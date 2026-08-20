"""Explicit Postgres chat-history read/write.

Why this module exists
----------------------
``setup_conversational_rag`` originally returned a ``RunnableWithMessageHistory``,
which persists turns through an ``on_end`` **listener** (``_exit_history``) rather
than as part of the chain. Callback exceptions are caught by
``langchain_core.callbacks.manager.handle_event``, logged at WARNING level and
re-raised only when ``handler.raise_error`` is true -- which it never is for
tracers. So a failed write produced a perfectly normal-looking answer and an
empty table.

On top of that, the SQL branch of ``smart_router_agent`` returns without ever
invoking the chain, so no listener configuration could have persisted those turns.

Doing the reads and writes here, on the main call path, means a Postgres failure
surfaces as a real error instead of vanishing, and both routes persist identically.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from api.config import get_settings

logger = logging.getLogger(__name__)


class HistoryUnavailable(RuntimeError):
    """The Postgres chat history could not be opened."""


@contextmanager
def open_history(session_key: str) -> Iterator[BaseChatMessageHistory]:
    """
    A context manager to open a `PostgresChatMessageHistory` instance and ensure its connection is properly closed.
    
    This function handles potential connection errors from `langchain_community`'s `PostgresChatMessageHistory`
    and translates them into a more actionable `HistoryUnavailable` exception.
    
    Args:
        session_key (str): The unique identifier for the chat session.
    
    Yields:
        Iterator[BaseChatMessageHistory]: An instance of `PostgresChatMessageHistory`.
    Raises:
        HistoryUnavailable: If a connection to PostgreSQL cannot be established or the history table is inaccessible."""
    """Open a ``PostgresChatMessageHistory`` and guarantee the socket is closed.

    ``langchain_community``'s implementation catches ``psycopg.OperationalError``
    in ``__init__``, logs it, and leaves ``self.cursor`` unset -- then immediately
    calls ``self.cursor.execute(...)``. The resulting ``AttributeError`` is
    translated here into something a caller can act on.

    Constructed directly rather than via ``retrieval5.get_postgres_session_history``
    so this module stays light. Importing retrieval5 from here would load it a
    second time under ``python retrieval5.py`` (once as ``__main__``, once as
    ``retrieval5``), duplicating the cross-encoder, router and SQLDatabase.
    """
    settings = get_settings()
    try:
        history = PostgresChatMessageHistory(
            connection_string=settings.postgres_dsn,
            session_id=session_key,
            table_name=settings.history_table,
        )
    except AttributeError as exc:
        raise HistoryUnavailable(
            "Could not connect to Postgres for chat history. Check POSTGRES_PWD / "
            "DB_NAME in creds.env and that the server is reachable."
        ) from exc

    if getattr(history, "connection", None) is None:
        raise HistoryUnavailable("Postgres chat history connection was not established.")

    try:
        yield history
    finally:
        # Release the socket deterministically rather than waiting for __del__,
        # which is unpredictable under a threadpool.
        try:
            history.cursor.close()
            history.connection.close()
        except Exception:  # noqa: BLE001 - closing must never mask the real error
            logger.debug("Failed to close chat history connection", exc_info=True)


def load_recent_messages(
    history: BaseChatMessageHistory, max_turns: int
) -> list[BaseMessage]:
    """
    Loads a specified number of recent chat messages from the history.
    This is used to keep the contextualizer prompt bounded.
    
    Args:
        history (BaseChatMessageHistory): The chat message history object.
        max_turns (int): The maximum number of conversational turns (user + AI messages) to retrieve.
    Returns:
        list[BaseMessage]: A list of the most recent `max_turns` messages.
    """
    """Last ``max_turns`` exchanges, so the contextualizer prompt stays bounded."""
    messages = history.messages
    if max_turns <= 0:
        return []
    return messages[-(2 * max_turns):]


def record_turn(history: BaseChatMessageHistory, question: str, answer: str) -> bool:
    """
    Records both the user's question and the assistant's answer to the chat history.
    This ensures that turns are persisted together after a successful answer,
    avoiding orphaned `HumanMessage` entries.
    
    Args:
        history (BaseChatMessageHistory): The chat message history object.
        question (str): The user's question.
        answer (str): The assistant's answer.
    Returns:
        bool: True if the turn was successfully persisted, False otherwise."""
    """Persist the user turn and the assistant turn together, after a successful answer.

    Both messages are written in one call. Writing the question up front would
    leave orphan ``HumanMessage`` rows behind whenever the LLM call fails, and the
    next turn's contextualizer would then rewrite against a dangling question.
    """
    try:
        history.add_messages([HumanMessage(content=question), AIMessage(content=answer)])
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist chat turn")
        return False
