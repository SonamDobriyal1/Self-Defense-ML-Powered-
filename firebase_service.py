from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

_USE_FIRESTORE = settings.firebase_enabled

if _USE_FIRESTORE:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.api_core import exceptions as gcloud_exceptions

    _firebase_app = None
    _firestore_client: Optional[firestore.Client] = None
else:
    firebase_admin = None
    firestore = None
    gcloud_exceptions = None
    _local_store_lock = Lock()
    _local_store_path = Path(settings.local_data_file).expanduser()
    _local_store: Dict[str, Any] = {"users": {}, "session_events": {}}
    _memory_users: Dict[str, Dict[str, Any]] = _local_store["users"]
    _memory_session_events: Dict[str, List[Dict[str, Any]]] = _local_store["session_events"]


def _ensure_local_store_loaded() -> None:
    if _USE_FIRESTORE:
        return
    with _local_store_lock:
        if _local_store_path.exists():
            try:
                data = json.loads(_local_store_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupted file
                logger.warning("Local data file is invalid JSON, starting fresh: %s", exc)
                data = {}
        else:
            _local_store_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}

        users = data.get("users", {})
        events = data.get("session_events", {})
        _local_store["users"].clear()
        _local_store["users"].update(users)
        _local_store["session_events"].clear()
        _local_store["session_events"].update(events)


def _persist_local_store() -> None:
    if _USE_FIRESTORE:
        return
    with _local_store_lock:
        payload = {
            "users": _local_store["users"],
            "session_events": _local_store["session_events"],
        }
        _local_store_path.parent.mkdir(parents=True, exist_ok=True)
        _local_store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if not _USE_FIRESTORE:
    _ensure_local_store_loaded()

SESSION_EVENTS_COLLECTION = "session_events"


def _firestore_timeout() -> float:
    return settings.firebase_timeout_seconds


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_sort_key(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def init_firebase() -> "firestore.Client":
    """
    Initialize Firebase Admin SDK using credentials supplied via configuration.
    """
    if not _USE_FIRESTORE:
        raise RuntimeError("Firestore backend is disabled (DATA_BACKEND!=firestore).")

    global _firebase_app, _firestore_client

    if _firestore_client is not None:
        return _firestore_client

    cred_source = settings.firebase_credentials
    if not cred_source:
        raise RuntimeError(
            "FIREBASE_CREDENTIALS is required when DATA_BACKEND=firestore. "
            "Provide a path to a service account JSON file or inline JSON string."
        )

    if os.path.isfile(cred_source):
        credentials_data = credentials.Certificate(cred_source)
    else:
        credentials_data = credentials.Certificate(json.loads(cred_source))

    _firebase_app = firebase_admin.initialize_app(credentials_data)
    _firestore_client = firestore.client(app=_firebase_app)
    logger.info("Firebase initialized successfully.")
    return _firestore_client


def _client() -> "firestore.Client":
    return init_firebase()


def _user_doc_id(email: str) -> str:
    sanitized = email.strip().lower()
    sanitized = sanitized.replace("@", "_at_").replace(".", "_dot_")
    return sanitized


def _user_doc_ref(email: str) -> "firestore.DocumentReference":
    if not _USE_FIRESTORE:
        raise RuntimeError("Firestore backend is disabled.")
    return _client().collection("users").document(_user_doc_id(email))


def _session_events_ref(email: str):
    if not _USE_FIRESTORE:
        raise RuntimeError("Firestore backend is disabled.")
    return _user_doc_ref(email).collection(SESSION_EVENTS_COLLECTION)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _memory_user(email: str) -> Dict[str, Any]:
    key = _normalize_email(email)
    return _memory_users.setdefault(
        key,
        {
            "email": key,
            "display_name": key.split("@")[0] if "@" in key else key,
            "password_hash": "",
            "role": "user",
            "progress": {
                "current_pose_index": 0,
                "completed_pose_ids": [],
                "last_updated": None,
            },
        },
    )


def get_user(email: str) -> Optional[Dict[str, Any]]:
    if not _USE_FIRESTORE:
        key = _normalize_email(email)
        user = _memory_users.get(key)
        if user:
            return {**user, "id": key}
        return None

    try:
        doc = _user_doc_ref(email).get(timeout=_firestore_timeout())
    except gcloud_exceptions.GoogleAPICallError as exc:
        logger.error("Firestore get_user failed for %s: %s", email, exc)
        raise RuntimeError("Unable to reach Firebase (get_user).") from exc

    if doc.exists:
        data = doc.to_dict()
        data["id"] = doc.id
        return data
    return None


def create_user(email: str, display_name: str, password_hash: str, role: str = "user") -> Dict[str, Any]:
    if not _USE_FIRESTORE:
        key = _normalize_email(email)
        payload = {
            "email": key,
            "display_name": display_name.strip() or key.split("@")[0],
            "password_hash": password_hash,
            "role": role,
            "progress": {
                "current_pose_index": 0,
                "completed_pose_ids": [],
                "last_updated": _now_iso(),
            },
        }
        _memory_users[key] = payload
        _persist_local_store()
        logger.info("Created local user %s with role %s", email, role)
        return payload

    doc_ref = _user_doc_ref(email)
    payload = {
        "email": email.strip().lower(),
        "display_name": display_name.strip(),
        "password_hash": password_hash,
        "role": role,
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "progress": {
            "current_pose_index": 0,
            "completed_pose_ids": [],
            "last_updated": firestore.SERVER_TIMESTAMP,
        },
    }
    try:
        doc_ref.set(payload, timeout=_firestore_timeout())
    except gcloud_exceptions.GoogleAPICallError as exc:
        logger.error("Firestore create_user failed for %s: %s", email, exc)
        raise RuntimeError("Unable to create user in Firebase.") from exc
    logger.info("Created user %s with role %s", email, role)
    return payload


def update_user_progress(
    email: str,
    *,
    current_pose_index: Optional[int] = None,
    completed_pose_id: Optional[int] = None,
    reset: bool = False,
) -> None:
    if not _USE_FIRESTORE:
        key = _normalize_email(email)
        user = _memory_users.setdefault(
            key,
            {
                "email": key,
                "display_name": key.split("@")[0],
                "password_hash": "",
                "role": "user",
                "progress": {
                    "current_pose_index": 0,
                    "completed_pose_ids": [],
                    "last_updated": None,
                },
            },
        )
        progress = user.setdefault(
            "progress",
            {
                "current_pose_index": 0,
                "completed_pose_ids": [],
                "last_updated": None,
            },
        )
        if reset:
            progress["current_pose_index"] = 0
            progress["completed_pose_ids"] = []
        else:
            if current_pose_index is not None:
                progress["current_pose_index"] = current_pose_index
            if completed_pose_id is not None and completed_pose_id not in progress["completed_pose_ids"]:
                progress["completed_pose_ids"].append(completed_pose_id)
        progress["last_updated"] = _now_iso()
        _persist_local_store()
        logger.info("Updated local progress for %s: %s", email, progress)
        return

    doc_ref = _user_doc_ref(email)
    updates: Dict[str, Any] = {"updated_at": firestore.SERVER_TIMESTAMP}

    if reset:
        updates.update({
            "progress.current_pose_index": 0,
            "progress.completed_pose_ids": [],
            "progress.last_updated": firestore.SERVER_TIMESTAMP,
        })
    else:
        if current_pose_index is not None:
            updates["progress.current_pose_index"] = current_pose_index
        if completed_pose_id is not None:
            updates["progress.completed_pose_ids"] = firestore.ArrayUnion([completed_pose_id])
        updates["progress.last_updated"] = firestore.SERVER_TIMESTAMP

    try:
        doc_ref.update(updates, timeout=_firestore_timeout())
    except gcloud_exceptions.GoogleAPICallError as exc:
        logger.error("Firestore update_user_progress failed for %s: %s", email, exc)
        raise RuntimeError("Unable to update progress in Firebase.") from exc
    logger.info("Updated progress for %s: %s", email, updates)


def list_users() -> List[Dict[str, Any]]:
    if not _USE_FIRESTORE:
        return [
            {**data, "id": key}
            for key, data in sorted(_memory_users.items())
        ]

    try:
        docs = _client().collection("users").stream(timeout=_firestore_timeout())
    except gcloud_exceptions.GoogleAPICallError as exc:
        logger.error("Firestore list_users failed: %s", exc)
        raise RuntimeError("Unable to fetch users from Firebase.") from exc
    results = []
    for doc in docs:
        entry = doc.to_dict()
        entry["id"] = doc.id
        results.append(entry)
    return results


def reset_progress(email: str) -> None:
    update_user_progress(email, reset=True)


def set_user_role(email: str, role: str) -> None:
    role = role.strip().lower()
    if not _USE_FIRESTORE:
        key = _normalize_email(email)
        user = _memory_users.get(key)
        if user:
            user["role"] = role
            _persist_local_store()
        return

    try:
        _user_doc_ref(email).update({"role": role}, timeout=_firestore_timeout())
    except gcloud_exceptions.GoogleAPICallError as exc:
        logger.error("Failed to update role for %s: %s", email, exc)
        raise RuntimeError("Unable to update user role.") from exc


def _build_session_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    event_id = payload.get("id") or uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc)
    base_event = {
        "id": event_id,
        "pose_id": payload.get("pose_id"),
        "pose_name": payload.get("pose_name"),
        "similarity": payload.get("similarity"),
        "status": payload.get("status", "matched"),
        "notes": payload.get("notes"),
        "captured_at": firestore.SERVER_TIMESTAMP if _USE_FIRESTORE else timestamp.isoformat(),
    }
    # Remove None values for cleaner storage
    return {k: v for k, v in base_event.items() if v is not None}


def append_session_event(email: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist a single training event so instructors can review recent matches.
    """
    event = _build_session_event(payload)
    if not _USE_FIRESTORE:
        key = _normalize_email(email)
        events = _memory_session_events.setdefault(key, [])
        events.append(event)
        _persist_local_store()
        logger.debug("Stored local session event for %s: %s", email, event)
        return event

    try:
        _session_events_ref(email).add(event, timeout=_firestore_timeout())
    except gcloud_exceptions.GoogleAPICallError as exc:
        logger.error("Failed to append session event for %s: %s", email, exc)
        raise RuntimeError("Unable to save session event.") from exc
    return event


def list_session_events(email: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Return the most recent training events for a user, newest first.
    """
    limit = max(1, min(limit, 50))
    if not _USE_FIRESTORE:
        key = _normalize_email(email)
        events = list(_memory_session_events.get(key, []))
        events.sort(key=lambda e: _event_sort_key(e.get("captured_at")), reverse=True)
        return events[:limit]

    try:
        query = (
            _session_events_ref(email)
            .order_by("captured_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        docs = query.stream(timeout=_firestore_timeout())
    except gcloud_exceptions.GoogleAPICallError as exc:
        logger.error("Failed to list session events for %s: %s", email, exc)
        raise RuntimeError("Unable to fetch session events.") from exc

    results: List[Dict[str, Any]] = []
    for doc in docs:
        entry = doc.to_dict()
        entry["id"] = doc.id
        results.append(entry)
    return results
