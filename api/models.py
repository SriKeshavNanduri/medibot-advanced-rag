"""Pydantic v2 request/response contracts for every endpoint.

Deliberately free of langchain/retrieval5 imports so the models can be imported
and unit-tested in isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.rbac import Role  # re-exported for convenience

__all__ = [
    "Role",
    "RouteName",
    "ComponentStatus",
    "LoginRequest",
    "LoginResponse",
    "ChatRequest",
    "Source",
    "SqlSource",
    "ChatResponse",
    "CollectionsResponse",
    "ComponentHealth",
    "HealthResponse",
    "ErrorResponse",
]


class RouteName(str, Enum):
    """Mirrors the semantic_router Route names in retrieval5.py."""

    QDRANT_RAG = "qdrant_rag"
    SQL_RAG = "sql_rag"


class ComponentStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


# ---------------------------------------------------------------- /login ---

class CreateUserRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"username": "dr_house", "password": "doctor123", "role": "doctor"}]
        },
    )

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256, repr=False)
    role: Role


class LoginRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"username": "dr_house", "password": "doctor123"}]
        },
    )

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256, repr=False)

    @field_validator("username")
    @classmethod
    def _strip_username(cls, value: str) -> str:
        """
        Strips whitespace from the username and ensures it's not blank.

        Args:
            value (str): The username string.
        Returns:
            str: The stripped username.
        Raises:
            ValueError: If the username is blank after stripping.
        """
        value = value.strip()
        if not value:
            raise ValueError("username must not be blank")
        return value


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds.")
    username: str
    role: Role
    accessible_collections: list[str] = Field(
        default_factory=list,
        description="Convenience echo of GET /collections/{role} for this token.",
    )


# ----------------------------------------------------------------- /chat ---
class ChatRequest(BaseModel):
    """There is deliberately no ``role`` field.

    The role is taken from the JWT ``role`` claim. ``extra="forbid"`` means a
    client that tries to smuggle ``{"role": "admin"}`` into the body gets a 422
    rather than a privilege escalation.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": "What is the policy for Sick Leave?", "session_id": "user:timestamp_randomHex"},
                {"question": "How many tickets are currently open?", "session_id": "user:timestamp_randomHex"},
            ]
        },
    )

    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:\-]+$",
        description=(
            "Client-chosen conversation id. Namespaced server-side with the "
            "authenticated username, so it cannot read another user's history."
        ),
    )

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        """
        Strips whitespace from the question and ensures it's not blank.

        Args:
            value (str): The question string.
        Returns:
            str: The stripped question.
        Raises:
            ValueError: If the question is blank after stripping.
        """
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Source(BaseModel):
    """One reranked chunk, built from the metadata written by chunking.py."""

    model_config = ConfigDict(extra="ignore")

    source_document: str = "unknown"
    collection: str | None = None
    section_title: str | None = None
    chunk_type: str | None = None
    snippet: str | None = Field(default=None, description="First ~240 chars of the chunk.")

    @classmethod
    def from_document(cls, doc: Any) -> "Source":
        """
        Creates a Source instance from a LangChain Document object.

        Args:
            doc (Any): A LangChain Document object.
        Returns:
            Source: A new Source instance populated with data from the document.
        """
        metadata = getattr(doc, "metadata", {}) or {}
        text = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        return cls(
            source_document=metadata.get("source_document") or "unknown",
            collection=metadata.get("collection"),
            section_title=metadata.get("section_title") or None,
            chunk_type=metadata.get("chunk_type"),
            snippet=(text[:240] + "…") if len(text) > 240 else (text or None),
        )


class SqlSource(BaseModel):
    """Provenance for the SQL RAG branch."""

    sql_query: str
    row_count: int | None = Field(
        default=None,
        description="None when the result string could not be parsed back into rows.",
    )


class ChatResponse(BaseModel):
    answer: str
    route: RouteName
    role: Role
    session_id: str
    sources: list[Source] = Field(default_factory=list)
    dense_docs: list[Source] = Field(default_factory=list)
    sql: SqlSource | None = None
    router_confidence: float | None = None
    history_written: bool = Field(
        description="True when both the user turn and the assistant turn were committed."
    )
    latency_ms: int


# ---------------------------------------------------- /collections/{role} ---
class CollectionsResponse(BaseModel):
    role: Role
    collections: list[str] = Field(
        description="Document categories (mediassist_data folders) readable by this role."
    )
    count: int

    @classmethod
    def build(cls, role: Role, collections: list[str]) -> "CollectionsResponse":
        """
        Constructs a CollectionsResponse instance.

        Args:
            role (Role): The role for which the collections are being listed.
            collections (list[str]): A list of collection names accessible to the role.
        Returns:
            CollectionsResponse: A new instance with sorted collections and their count.
        """
        ordered = sorted(collections)
        return cls(role=role, collections=ordered, count=len(ordered))


# --------------------------------------------------------------- /health ---
class ComponentHealth(BaseModel):
    status: ComponentStatus
    detail: str | None = None
    latency_ms: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    postgres: ComponentHealth
    qdrant: ComponentHealth
    chains_ready: list[Role] = Field(default_factory=list)
    history_table: str | None = None
    history_table_exists: bool | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------- errors ---
class ErrorResponse(BaseModel):
    error: str = Field(description="Short machine-readable code.")
    detail: str | None = None
    status_code: int
