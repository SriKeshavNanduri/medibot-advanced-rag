"""Configuration for the MediBot API.

Loads ``creds.env`` from an **absolute** path derived from ``__file__``.

This matters: ``retrieval5.py`` and ``chunking.py`` both call
``load_dotenv("creds.env")`` with a *relative* path, so launching uvicorn from
any directory other than the project root leaves every ``os.getenv`` returning
``None``. In particular ``CONTEXT_TABLE`` becomes ``None``, and
``PostgresChatMessageHistory`` then happily runs ``CREATE TABLE IF NOT EXISTS
None (...)`` -- rows land in a table literally called ``none`` while the table
you are inspecting stays empty. Loading from an absolute path here, before
anything imports ``retrieval5``, removes that failure mode.
"""

from __future__ import annotations

import os
import urllib.parse
import warnings
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDS_ENV = PROJECT_ROOT / "creds.env"

load_dotenv(CREDS_ENV, override=False)

DEV_JWT_SECRET = "dev-only-insecure-secret-change-me"


class Settings:
    """Plain settings object; values are read once at import."""

    api_title: str = "MediBot RAG API"
    api_version: str = "1.0.0"

    # --- auth ---
    jwt_secret: str = os.getenv("JWT_SECRET", DEV_JWT_SECRET)
    jwt_algorithm: str = "HS256"
    jwt_ttl_seconds: int = int(os.getenv("JWT_TTL_SECONDS", str(8 * 60 * 60)))
    jwt_issuer: str = "medibot-api"
    jwt_audience: str = "medibot-client"

    # --- datastores ---
    db_name: str | None = os.getenv("DB_NAME")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "5432")
    qdrant_url: str | None = os.getenv("LOCAL_QDRANT_URL")
    qdrant_collection: str | None = os.getenv("COLLECTION_NAME")
    history_table: str | None = os.getenv("CONTEXT_TABLE")
    users_table: str = os.getenv("USERS_TABLE", "app_users")

    # --- tuning ---
    history_max_turns: int = int(os.getenv("HISTORY_MAX_TURNS", "10"))
    max_concurrent_chats: int = int(os.getenv("MAX_CONCURRENT_CHATS", "4"))

    @property
    def postgres_dsn(self) -> str:
        """
        Constructs the PostgreSQL connection string (DSN) from environment variables.
        
        Returns:
            str: The PostgreSQL connection string."""
        """libpq/psycopg connection string (same shape as retrieval5.py:169)."""
        pwd = urllib.parse.quote_plus(str(os.getenv("POSTGRES_PWD")))
        return (
            f"postgresql://{self.db_user}:{pwd}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retrieves and caches the application settings.
    This function ensures that settings are loaded only once.
    
    Returns:
        Settings: An instance of the Settings class populated with configuration values.
    Raises:
        RuntimeError: If critical environment variables (CONTEXT_TABLE, DB_NAME, QDRANT_URL/COLLECTION_NAME) are not set.
    """
    settings = Settings()

    if not settings.history_table:
        raise RuntimeError(
            f"CONTEXT_TABLE is not set. Check that {CREDS_ENV} exists and defines it. "
            "Without it the chat history table would be named 'none'."
        )
    if not settings.db_name:
        raise RuntimeError(f"DB_NAME is not set. Check {CREDS_ENV}.")
    if not settings.qdrant_url or not settings.qdrant_collection:
        raise RuntimeError(f"LOCAL_QDRANT_URL / COLLECTION_NAME not set. Check {CREDS_ENV}.")
    if settings.jwt_secret == DEV_JWT_SECRET:
        warnings.warn(
            "JWT_SECRET is not set; using an insecure development secret. "
            "Set JWT_SECRET in the environment before exposing this service.",
            stacklevel=2,
        )

    return settings
