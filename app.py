import eventlet  # Required for Flask-SocketIO server, no global monkey-patching to keep gRPC happy.

import base64
import logging
import os
import socket
from datetime import datetime
from functools import wraps
from typing import Dict, Optional

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_socketio import SocketIO, emit
from werkzeug.security import check_password_hash, generate_password_hash

from config import settings
from firebase_service import (
    append_session_event,
    create_user,
    get_user,
    list_session_events,
    list_users,
    reset_progress,
    set_user_role,
    update_user_progress,
)
from pose_detector import PoseDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path="/static")
app.config["SECRET_KEY"] = settings.secret_key
app.config["ENV"] = settings.flask_env
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "static/poses"
)

# Create static/poses directory if it doesn't exist
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Initialize pose detector with higher threshold
pose_detector = PoseDetector(similarity_threshold=90.0)


def load_tutorial_poses():
    poses = []
    if os.path.exists(app.config["UPLOAD_FOLDER"]):
        for img_file in sorted(os.listdir(app.config["UPLOAD_FOLDER"])):
            if img_file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(app.config["UPLOAD_FOLDER"], img_file)
                keypoints = pose_detector.get_pose_landmarks(img_path)
                if keypoints is not None and len(keypoints) > 0:
                    poses.append(
                        {
                            "landmarks": keypoints.copy(),
                            "filename": img_file,
                        }
                    )
                    logger.info("Loaded pose image: %s", img_file)
    return poses


# Initialize variables
tutorial_poses = []

# Track per-user pose index inside the live session
user_pose_state: Dict[str, Dict[str, int]] = {}

# Load poses when the application starts
with app.app_context():
    tutorial_poses = load_tutorial_poses()


def get_logged_in_email() -> Optional[str]:
    return session.get("user_email")


def ensure_user_pose_state(email: str) -> Dict[str, int]:
    state = user_pose_state.setdefault(email, {"current_pose_index": 0})
    total = max(len(tutorial_poses) - 1, 0)
    state["current_pose_index"] = max(0, min(state["current_pose_index"], total))
    return state


def fetch_user_record(email: str) -> Optional[dict]:
    if not email:
        return None
    try:
        return get_user(email)
    except Exception as exc:  # pragma: no cover - connectivity issues
        logger.error("Failed to fetch user %s: %s", email, exc)
        return None


def ensure_default_admin_user():
    email = settings.admin_email
    password = settings.admin_password
    if not email or not password:
        logger.info("ADMIN_EMAIL or ADMIN_PASSWORD not set; skip seeding admin user.")
        return

    display_name = settings.admin_name or "Guardian Admin"
    user = fetch_user_record(email)
    if user is None:
        try:
            hashed = generate_password_hash(password)
            create_user(email, display_name, hashed, role="admin")
            logger.info("Created default admin user %s", email)
        except Exception as exc:
            logger.error("Failed to create default admin user %s: %s", email, exc)
    else:
        if user.get("role") != "admin":
            try:
                set_user_role(email, "admin")
                logger.info("Promoted %s to admin role.", email)
            except Exception as exc:
                logger.error("Failed to promote %s to admin role: %s", email, exc)


ensure_default_admin_user()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not get_logged_in_email():
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)

    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if session.get("user_role") != "admin":
            abort(403)
        return fn(*args, **kwargs)

    return wrapper


@app.before_request
def inject_user():
    g.user_email = get_logged_in_email()
    g.display_name = session.get("display_name")


@app.context_processor
def template_globals():
    return {
        "current_user_email": session.get("user_email"),
        "current_user_name": session.get("display_name"),
        "current_user_role": session.get("user_role"),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        total_poses=len(tutorial_poses),
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        display_name = request.form.get("display_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not display_name or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        try:
            if get_user(email):
                flash("An account with that email already exists.", "error")
                return redirect(url_for("signup"))

            password_hash = generate_password_hash(password)
            create_user(email, display_name, password_hash)
            session["user_email"] = email
            session["display_name"] = display_name
            session["user_role"] = "user"
            flash("Welcome to the academy! Let's get training.", "success")
            return redirect(url_for("dashboard"))
        except Exception as exc:  # pragma: no cover - connectivity issues
            logger.exception("Signup failed: %s", exc)
            flash(
                "Unable to complete signup. Verify Firebase credentials and try again.",
                "error",
            )
            return redirect(url_for("signup"))

    return render_template("auth_signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please provide both email and password.", "error")
            return redirect(url_for("login"))

        user_record = fetch_user_record(email)
        if not user_record:
            flash("Account not found. Please sign up first.", "error")
            return redirect(url_for("login"))

        password_hash = user_record.get("password_hash")
        if not password_hash or not check_password_hash(password_hash, password):
            flash("Incorrect password. Try again.", "error")
            return redirect(url_for("login"))

        session["user_email"] = email
        session["display_name"] = user_record.get("display_name")
        session["user_role"] = user_record.get("role", "user")
        flash("Welcome back! Keep pushing your limits.", "success")
        next_url = request.args.get("next")
        return redirect(next_url or url_for("dashboard"))

    return render_template("auth_login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out safely.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    email = get_logged_in_email()
    user_record = fetch_user_record(email) or {}
    progress = user_record.get("progress", {})
    completed = len(progress.get("completed_pose_ids", []))
    total = max(len(tutorial_poses), 1)
    percent = int(min(100, (completed / total) * 100))

    recent_events = []
    try:
        events = list_session_events(email, limit=5)
        for evt in events:
            ts = evt.get("captured_at")
            ts_label = "--"
            if isinstance(ts, datetime):
                ts_label = ts.strftime("%b %d · %H:%M")
            elif hasattr(ts, "isoformat"):
                try:
                    ts_label = ts.isoformat(timespec="minutes")
                except TypeError:
                    ts_label = ts.isoformat()
            elif isinstance(ts, str):
                ts_label = ts
            recent_events.append(
                {
                    "pose": evt.get("pose_name") or f"Pose {evt.get('pose_id')}",
                    "similarity": int(round(evt.get("similarity") or 0)),
                    "status": (evt.get("status") or "matched").title(),
                    "captured_at": ts_label,
                }
            )
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Session event lookup skipped: %s", exc)

    return render_template(
        "dashboard.html",
        user=user_record,
        progress=progress,
        percent_complete=percent,
        total_poses=len(tutorial_poses),
        recent_events=recent_events,
    )


@app.route("/tutorial")
@login_required
def tutorial():
    if not tutorial_poses:
        flash(
            "No training poses found. Add reference images to static/poses to begin.",
            "warning",
        )
        return redirect(url_for("dashboard"))

    email = get_logged_in_email()
    user_record = fetch_user_record(email) or {}
    progress = user_record.get("progress", {})
    starting_index = int(progress.get("current_pose_index", 0))
    total = len(tutorial_poses)
    starting_index = max(0, min(starting_index, total - 1))

    poses_meta = [
        {"filename": pose["filename"], "index": idx} for idx, pose in enumerate(tutorial_poses)
    ]

    video_folder = os.path.join(app.static_folder, "poses_video")
    tutorial_video_count = (
        len(os.listdir(video_folder)) if os.path.exists(video_folder) else len(tutorial_poses)
    )

    return render_template(
        "tutorial.html",
        total_poses=total,
        poses_meta=poses_meta,
        starting_index=starting_index,
        tutorial_videos=tutorial_video_count,
    )


@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    try:
        users = list_users()
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load users: %s", exc)
        users = []

    total_users = len(users)
    total_poses = max(len(tutorial_poses), 1)

    def progress_stats(user):
        progress = user.get("progress", {}) if isinstance(user, dict) else getattr(user, "progress", {})
        current_pose = int(progress.get("current_pose_index", 0) if isinstance(progress, dict) else 0)
        completed = progress.get("completed_pose_ids", []) if isinstance(progress, dict) else []
        pct = int(min(100, (current_pose / total_poses) * 100))
        return current_pose, len(completed), pct

    dashboard_rows = []
    total_completion_pct = 0
    total_completed_count = 0
    for user in users:
        current_pose, completed_count, pct = progress_stats(user)
        total_completion_pct += pct
        total_completed_count += completed_count
        dashboard_rows.append({
            "display_name": user.get("display_name", "Unknown"),
            "email": user.get("email"),
            "role": user.get("role", "user"),
            "current_pose": current_pose,
            "completed_count": completed_count,
            "progress_pct": pct,
        })

    avg_completion = int(total_completion_pct / total_users) if total_users else 0

    return render_template(
        "admin.html",
        users=dashboard_rows,
        total_poses=len(tutorial_poses),
        summary={
            "total_users": total_users,
            "avg_completion": avg_completion,
            "total_completed_poses": total_completed_count,
        },
    )


@app.route("/progress/reset", methods=["POST"])
@login_required
def progress_reset():
    email = get_logged_in_email()
    try:
        reset_progress(email)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to reset progress: %s", exc)
        flash("Could not reset progress. Please try again.", "error")
        return redirect(url_for("dashboard"))

    ensure_user_pose_state(email)["current_pose_index"] = 0
    wants_json = request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]
    if wants_json:
        return jsonify({"success": True})

    flash("Progress cleared. Start fresh when you're ready.", "success")
    return redirect(url_for("dashboard"))


@app.route("/progress/update", methods=["POST"])
@login_required
def progress_update():
    email = get_logged_in_email()
    data = request.get_json(force=True, silent=True) or {}
    pose_index = int(data.get("current_pose_index", 0))
    completed_pose_id = data.get("completed_pose_id")

    pose_index = max(0, min(pose_index, max(len(tutorial_poses) - 1, 0)))

    try:
        update_user_progress(
            email,
            current_pose_index=pose_index,
            completed_pose_id=completed_pose_id,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to update progress: %s", exc)
        return jsonify({"success": False, "error": "Progress update failed"}), 500

    ensure_user_pose_state(email)["current_pose_index"] = pose_index
    return jsonify({"success": True, "next_pose_index": pose_index})


@app.route("/update_pose_index/<int:index>", methods=["POST"])
@login_required
def update_pose_index(index):
    email = get_logged_in_email()
    if index < 0 or index >= len(tutorial_poses):
        return jsonify({"success": False, "error": "Invalid index"}), 400

    state = ensure_user_pose_state(email)
    state["current_pose_index"] = index

    try:
        update_user_progress(email, current_pose_index=index)
    except Exception as exc:  # pragma: no cover
        logger.warning("Pose index update stored locally only: %s", exc)

    # Reset detector state for the new pose
    pose_detector.highest_similarity = 0.0
    pose_detector.best_frame = None
    if hasattr(handle_frame, "match_counter"):
        handle_frame.match_counter = 0

    return jsonify({"success": True, "current_index": index})


@app.route("/check_pose_match")
@login_required
def check_pose_match():
    email = get_logged_in_email()
    state = ensure_user_pose_state(email)
    index = state.get("current_pose_index", 0)
    if 0 <= index < len(tutorial_poses):
        if pose_detector.current_keypoints is not None:
            similarity = pose_detector.calculate_pose_similarity(
                pose_detector.current_keypoints,
                tutorial_poses[index]["landmarks"],
            )
            similarity = float(similarity)
            if similarity >= pose_detector.similarity_threshold:
                return jsonify({"matched": True, "similarity": similarity})
    return jsonify({"matched": False, "similarity": 0.0})


def process_image(image_data, pose_index: int):
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode image")
            return None

        if 0 <= pose_index < len(tutorial_poses):
            frame, _, _, _ = pose_detector.process_frame(
                frame,
                tutorial_poses[pose_index]["landmarks"],
            )
        else:
            frame, _, _, _ = pose_detector.process_frame(frame)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded_image}"
    except Exception as e:
        logger.error("Error processing image: %s", e)
        return None


@socketio.on("connect")
def handle_connect():
    email = get_logged_in_email()
    if not email:
        emit("error", {"message": "Authentication required"})
        return False

    ensure_user_pose_state(email)
    emit("connection_test", {"message": "WebSocket connection successful!"})


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected: %s", request.sid)


@socketio.on("frame")
def handle_frame(data):
    email = get_logged_in_email()
    if not email:
        emit("error", {"message": "Authentication required"})
        return

    state = ensure_user_pose_state(email)
    pose_index = state.get("current_pose_index", 0)

    try:
        if not hasattr(handle_frame, "frame_counter"):
            handle_frame.frame_counter = 0

        handle_frame.frame_counter += 1
        if handle_frame.frame_counter % 3 != 0:
            return

        if not isinstance(data, dict) or "image" not in data:
            logger.error("Invalid data payload for frame event.")
            return

        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Error decoding frame from client.")
            return

        reference_keypoints = None
        if 0 <= pose_index < len(tutorial_poses):
            reference_keypoints = tutorial_poses[pose_index].get("landmarks")

        processed_frame, similarity, has_person, _ = pose_detector.process_frame(
            frame,
            reference_keypoints,
        )

        _, buffer = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        processed_base64 = base64.b64encode(buffer).decode("utf-8")
        emit("processed_frame", {"image": f"data:image/jpeg;base64,{processed_base64}"})

        if pose_detector.current_keypoints is not None and reference_keypoints is not None:
            emit(
                "pose_similarity",
                {
                    "similarity": float(similarity),
                    "pose_id": pose_index + 1,
                    "pose_name": tutorial_poses[pose_index]["filename"],
                },
            )

            threshold = float(pose_detector.similarity_threshold)
            if float(similarity) >= threshold:
                emit(
                    "pose_match_confirmed",
                    {
                        "matched": True,
                        "pose_id": pose_index + 1,
                        "pose_name": tutorial_poses[pose_index]["filename"],
                        "similarity": float(similarity),
                        "message": f"Excellent! You matched {tutorial_poses[pose_index]['filename']} with {similarity:.1f}% accuracy!",
                    },
                )

                next_index = min(pose_index + 1, max(len(tutorial_poses) - 1, 0))
                state["current_pose_index"] = next_index
                try:
                    update_user_progress(
                        email,
                        current_pose_index=next_index,
                        completed_pose_id=pose_index + 1,
                    )
                    append_session_event(
                        email,
                        {
                            "pose_id": pose_index + 1,
                            "pose_name": tutorial_poses[pose_index]["filename"],
                            "similarity": float(similarity),
                            "status": "matched",
                        },
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Progress update skipped: %s", exc)

        elif not has_person:
            emit(
                "pose_similarity",
                {"similarity": 0, "message": "No person detected"},
            )

    except Exception as e:  # pragma: no cover - diagnostics
        logger.exception("Error processing frame: %s", e)
        emit("error", {"message": "Error processing frame"})


def generate_frames():
    """Legacy function - kept for compatibility but not used in WebSocket version"""
    pass


@app.route("/video_feed")
def video_feed():
    """Legacy route - kept for compatibility but not used in WebSocket version"""
    return Response("WebSocket mode active", mimetype="text/plain")


if __name__ == "__main__":
    env_port = os.environ.get("PORT")

    def pick_open_port(base_port: int, attempts: int = 20) -> int:
        for p in range(base_port, base_port + attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]

    default_base_port = settings.port or 10000

    if env_port:
        port = int(env_port)
        try:
            socketio.run(app, host="0.0.0.0", port=port, debug=False)
        except OSError as e:
            logger.error("Port %s is busy. Stop the other process or choose a different PORT.", port)
            raise
    else:
        port = pick_open_port(default_base_port)
        if port != default_base_port:
            logger.info("Port %s busy; using available port %s instead.", default_base_port, port)
        socketio.run(app, host="0.0.0.0", port=port, debug=False)
