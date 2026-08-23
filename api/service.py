"""Per-role chain cache, request orchestration and health probes."""

from __future__ import annotations

import logging
import threading
import time

from api.config import get_settings
from api.history import load_recent_messages, open_history, record_turn
from api.models import ComponentHealth, ComponentStatus, Role
from api.rbac import get_collections_for_role, known_roles
import api.retrieval as retrieval  # cross-encoder, semantic router sync, SQLDatabase reflection

logger = logging.getLogger(__name__)

#: role value -> bare conversational RAG chain
CHAIN_CACHE: dict[str, object] = {}

# The cross-encoder and BM25 model are process-wide singletons shared by every
# cached chain. Sync endpoints run on AnyIO's 40-thread pool, and 40 concurrent
# reranker passes plus 40 psycopg connections would exhaust memory or Postgres
# max_connections. Cap the real concurrency here.
_CHAT_SEMAPHORE = threading.BoundedSemaphore(get_settings().max_concurrent_chats)


class ServiceNotReady(RuntimeError):
    """No chain is available for the requested role."""


def warmup() -> None:
    """
    Performs the initial setup and warm-up for the service.
    This involves importing heavy modules, building RAG chains for each role,
    and logging the completion status.
    
    This function is blocking and is intended to be called during the application's startup.
    
    Side Effects:
        - Imports `retrieval5` module.
        - Populates the `CHAIN_CACHE` dictionary with RAG chains for each role.
        - Logs the time taken for warmup and the status of each role's chain.
    
    Raises:
        RuntimeError: If a role is present in the configuration but a chain could not be built for it.
    """
    started = time.perf_counter()

    roles = known_roles()
    drift = set(roles) ^ {r.value for r in Role}
    if drift:
        logger.warning("Role enum / ACCESS_MAPPING drift: %s", sorted(drift))

    for role in roles:
        role_started = time.perf_counter()
        collections = get_collections_for_role(role)
        CHAIN_CACHE[role] = retrieval.setup_conversational_rag(
            user_role=role,
            allowed_collections=collections,
        )
        logger.info(
            "chain ready: %-18s collections=%s (%.2fs)",
            role,
            collections,
            time.perf_counter() - role_started,
        )

    logger.info("warmup finished in %.2fs", time.perf_counter() - started)


def ready_roles() -> list[Role]:
    return [Role(role) for role in sorted(CHAIN_CACHE)]


def answer_question(question: str, role: Role, session_key: str) -> dict:
    """Blocking; called from a sync endpoint, i.e. on a threadpool worker.

    Ordering matters:
      1. open one Postgres connection for the whole turn
      2. load prior messages BEFORE the LLM call -- the contextualizer in
         retrieval5 needs them to rewrite follow-ups into standalone questions
      3. answer
      4. persist both messages only after a successful answer, so a failed turn
         leaves history exactly as it was rather than stranding an orphan question
    """

    chain = CHAIN_CACHE.get(role.value)
    if chain is None:
        raise ServiceNotReady(f"No chain built for role {role.value}; service not warmed up.")

    started = time.perf_counter()
    with _CHAT_SEMAPHORE:
        with open_history(session_key) as history:
            result = retrieval.smart_router_agent(
                question=question,
                session_id=session_key,
                chain=chain,
                verbose=True,
                role=role,
            )
            # Runs for the qdrant_rag AND sql_rag branches alike -- the SQL path
            # previously bypassed the chain entirely and never wrote anything.
            result["history_written"] = record_turn(history, question, result["answer"])

    result["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return result


# ------------------------------------------------------------- health ---
def check_postgres() -> tuple[ComponentHealth, bool | None]:
    """Ping Postgres and report whether the chat history table exists."""
    """
    Checks the health of the PostgreSQL database and verifies the existence of the chat history table.
    
    Returns:
        tuple[ComponentHealth, bool | None]: A tuple containing:
            - ComponentHealth: The health status of the PostgreSQL component.
            - bool | None: True if the history table exists, False if not, or None if the check failed.
    """

    import psycopg

    settings = get_settings()
    started = time.perf_counter()
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute("SELECT to_regclass(%s)", (settings.history_table,))
                table_exists = cur.fetchone()[0] is not None
        elapsed = int((time.perf_counter() - started) * 1000)
        return ComponentHealth(status=ComponentStatus.OK, latency_ms=elapsed), table_exists
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(status=ComponentStatus.ERROR, detail=str(exc)[:300]), None


def check_qdrant() -> ComponentHealth:
    """
    Checks the health of the Qdrant vector database and retrieves collection information.
    
    Returns:
        ComponentHealth: The health status of the Qdrant component, including
                         details about the collection if successful.
    """

    from qdrant_client import QdrantClient

    settings = get_settings()
    started = time.perf_counter()
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=5)
        info = client.get_collection(settings.qdrant_collection)
        elapsed = int((time.perf_counter() - started) * 1000)
        return ComponentHealth(
            status=ComponentStatus.OK,
            latency_ms=elapsed,
            detail=f"collection={settings.qdrant_collection} points={info.points_count}",
        )
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(status=ComponentStatus.ERROR, detail=str(exc)[:300])
