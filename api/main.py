"""MediBot RAG API.

Run from the project root:

    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

Endpoints are sync ``def`` on purpose. Everything downstream blocks -- the chain
invoke, the cross-encoder forward pass, psycopg, and db.run() -- so FastAPI's
threadpool is the right place for them; ``async def`` would stall the event loop
and serialise every request in the process, including /health.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import api.chunking as chunking_service
from api import service as service
from api.auth import CurrentUserDep, authenticate, create_access_token
from api.config import get_settings
from api.db import count_users, create_user, ensure_users_table
from api.history import HistoryUnavailable
from api.models import (
    ChatRequest,
    ChatResponse,
    CollectionsResponse,
    ErrorResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    CreateUserRequest,
    Role,
    RouteName,
    Source,
    SqlSource,
)
from api.rbac import get_collections_for_role

import torch
torch._dynamo.config.suppress_errors = True

from dotenv import load_dotenv
load_dotenv("creds.env")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("medibot.api")

_settings = get_settings()

COMMON_ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing, expired or invalid token"},
    403: {"model": ErrorResponse, "description": "Role not permitted"},
    503: {"model": ErrorResponse, "description": "A backing service is unavailable"},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI application, including startup and shutdown events.
    Ensures the user table exists, warns if no users are present, and warms up services.
    """
    ensure_users_table()
    if count_users() == 0:
        logger.warning("No rows in %s -- run `python seed_users.py`.", _settings.users_table)

    logger.info("Warming up (cross-encoder, BM25, semantic router, one chain per role)...")
    await anyio.to_thread.run_sync(service.warmup)
    logger.info("Ready. Roles: %s", [r.value for r in service.ready_roles()])
    yield
    service.CHAIN_CACHE.clear()


app = FastAPI(
    title=_settings.api_title,
    version=_settings.api_version,
    description="Role-aware hybrid RAG + SQL RAG over the MediAssist corpus.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles HTTPException instances, converting them into a standardized ErrorResponse.
    
    Args:
        request (Request): The incoming request.
        exc (HTTPException): The exception that was raised.
    
    Returns:
        JSONResponse: A JSON response with a standardized error format."""
    body = ErrorResponse(
        error="internal_error" if exc.status_code >= 500 else str(exc.detail)[:80],
        detail=str(exc.detail),
        status_code=exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles RequestValidationError instances, converting them into a standardized ErrorResponse.
    This provides a cleaner and more consistent error message for validation failures.
    
    Args:
        request (Request): The incoming request.
        exc (RequestValidationError): The validation exception that was raised.
    
    Returns:
        JSONResponse: A JSON response with a standardized error format for validation errors."""
    body = ErrorResponse(
        error="validation_error",
        detail="; ".join(
            f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()
        ),
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )
    return JSONResponse(status_code=body.status_code, content=body.model_dump())


# ---------------------------------------------------------------- routes ---
@app.post(
    "/login",
    response_model=LoginResponse,
    tags=["auth"],
    summary="Exchange credentials for a role-tagged session token",
    responses={401: COMMON_ERRORS[401]},
)

def login(payload: LoginRequest) -> LoginResponse:
    """
    Authenticates a user and issues a JWT token.
    
    Args:
        payload (LoginRequest): The login credentials containing username and password.
    
    Returns:
        LoginResponse: A response containing the access token, its expiration,
                       the username, role, and accessible collections.
    
    Raises:
        HTTPException: If authentication fails due to invalid credentials."""
    
    role = authenticate(payload.username, payload.password)
    if role is None:
        # Identical message for unknown user and wrong password.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, ttl = create_access_token(payload.username, role)
    logger.info("login ok user=%s role=%s", payload.username, role.value)
    return LoginResponse(
        access_token=token,
        expires_in=ttl,
        username=payload.username,
        role=role,
        accessible_collections=get_collections_for_role(role.value),
    )


@app.post(
    "/createusers",
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
    summary="Create a user by username and password",
)
def create_user_route(payload: CreateUserRequest) -> dict[str, str]:
    """Create a user using the existing PBKDF2 password hashing flow."""
    created = create_user(payload.username, payload.password)
    if not created:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"User '{payload.username}' already exists.",
        )
    logger.info("user created username=%s", payload.username)
    return {
        "username": payload.username,
        "message": "User created successfully.",
    }


@app.post(
    "/ingestion",
    tags=["ingestion"],
    summary="Upload a PDF, Markdown or TXT document and ingest it using the existing chunking pipeline",
)
async def ingestion(
    file: UploadFile = File(...),
    collection: str = "general",
) -> dict[str, object]:
    """Ingest a single uploaded document using the existing Docling + chunking flow."""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    allowed_suffixes = {".pdf", ".md", ".txt"}
    allowed_collections = ["general", "billing", "equipment", "nursing", "clinical"]

    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported file format. Only .pdf, .md, and .txt files are allowed "
                f"for ingestion; received '{filename}'."
            ),
        )

    collection_name = collection.strip().lower()
    if collection_name not in allowed_collections:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid collection. Allowed values are: "
                + ", ".join(allowed_collections)
                + f". Received '{collection}'."
            ),
        )

    temp_path = None
    try:
        chunking_service.ensure_chunker_initialized()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        result = chunking_service._process_single_file(temp_path, collection_name)

        return result

    except Exception as e:
        return result 


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["rag"],
    summary="Ask a question; RBAC-filtered hybrid+rerank RAG or SQL RAG",
    responses=COMMON_ERRORS,
)

def chat(payload: ChatRequest, user: CurrentUserDep) -> ChatResponse:
    """
    Processes a chat question using either RAG or SQL RAG based on semantic routing.
    The user's role from the JWT token is used for access control.
    
    Args:
        payload (ChatRequest): The chat request containing the question and session ID.
        user (CurrentUserDep): The authenticated user object, injected by FastAPI's dependency.
    
    Returns:
        ChatResponse: The response containing the answer, route taken, sources, SQL query (if applicable),
                      router confidence, and latency.
    
    Raises:
        HTTPException: If the service is unavailable or a RAG pipeline error occurs."""
    # RBAC: the role comes from the JWT claim, never from the request body.
    # ChatRequest has no `role` field and forbids extras, so a client attempting
    # {"role": "admin"} gets a 422.
    session_key = user.session_key(payload.session_id)

    try:
        result = service.answer_question(payload.question, user.role, session_key)
    except (HistoryUnavailable, service.ServiceNotReady) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed user=%s role=%s", user.username, user.role.value)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"RAG pipeline error: {type(exc).__name__}: {exc}"
        )

    sql = None
    if result["route"] == RouteName.SQL_RAG.value and result.get("sql_query"):
        sql = SqlSource(sql_query=result["sql_query"], row_count=result.get("row_count"))

    documents = result["documents"]
    dense_docs = result.get("dense_docs", [])
    source_dense_docs = [Source.from_document(d) for d in dense_docs]

    return ChatResponse(
        answer=result["answer"],
        route=RouteName(result["route"]),
        dense_docs=source_dense_docs,
        role=user.role,
        session_id=payload.session_id,
        sources=[Source.from_document(d) for d in documents],
        sql=sql,
        router_confidence=result.get("confidence"),
        history_written=result.get("history_written", False),
        latency_ms=result.get("latency_ms", 0),
    )


@app.get(
    "/collections/{role}",
    response_model=CollectionsResponse,
    tags=["rbac"],
    summary="Document collections accessible to a role",
    responses=COMMON_ERRORS,
)

def list_collections(role: Role, user: CurrentUserDep) -> CollectionsResponse:
    """
    Retrieves the document collections accessible to a specified role.
    
    Args:
        role (Role): The role for which to retrieve accessible collections.
        user (CurrentUserDep): The authenticated user object.
    
    Returns:
        CollectionsResponse: A response containing the role and a list of accessible collection names.
    
    Raises:
        HTTPException: If the authenticated user's role does not permit querying the specified role."""
    # `role: Role` makes FastAPI reject an unknown role with a 422 listing the
    # valid values -- no handwritten validation needed.
    if user.role is not Role.ADMIN and role is not user.role:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Token role '{user.role.value}' may not query role '{role.value}'.",
        )
    return CollectionsResponse.build(role, get_collections_for_role(role.value))


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness and dependency check",
)

def health() -> HealthResponse:
    """
    Performs liveness and dependency checks for the API.
    
    Returns:
        HealthResponse: A response indicating the overall health status, version,
                        and the health of PostgreSQL, Qdrant, and chain readiness."""
    postgres, table_exists = service.check_postgres()
    qdrant = service.check_qdrant()
    roles = service.ready_roles()

    healthy = (
        postgres.status.value == "ok"
        and qdrant.status.value == "ok"
        and len(roles) == len(Role)
    )
    # Always HTTP 200 so probes can read the body and see which part is down.
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=_settings.api_version,
        postgres=postgres,
        qdrant=qdrant,
        chains_ready=roles,
        history_table=_settings.history_table,
        history_table_exists=table_exists,
    )
