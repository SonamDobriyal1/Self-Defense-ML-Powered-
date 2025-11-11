from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Ensure .env is loaded before settings are accessed anywhere else
load_dotenv()


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    flask_env: str
    secret_key: str
    port: int
    data_backend: str
    local_data_file: str
    firebase_credentials: Optional[str]
    firebase_timeout_seconds: float
    admin_email: Optional[str]
    admin_password: Optional[str]
    admin_name: Optional[str]

    @property
    def firebase_enabled(self) -> bool:
        return self.data_backend == "firestore"


def load_settings() -> Settings:
    return Settings(
        flask_env=_env("FLASK_ENV", "production"),
        secret_key=_env("SECRET_KEY", "change-me"),
        port=int(_env("PORT", "5000")),
        data_backend=_env("DATA_BACKEND", "memory").strip().lower(),
        local_data_file=_env("LOCAL_DATA_FILE", os.path.join("data", "local_store.json")),
        firebase_credentials=_env("FIREBASE_CREDENTIALS"),
        firebase_timeout_seconds=float(_env("FIREBASE_TIMEOUT_SECONDS", "30")),
        admin_email=_env("ADMIN_EMAIL"),
        admin_password=_env("ADMIN_PASSWORD"),
        admin_name=_env("ADMIN_NAME"),
    )


settings = load_settings()
