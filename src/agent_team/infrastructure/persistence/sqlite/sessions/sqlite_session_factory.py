"""Agents SDK SQLite session factory."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agents.memory import Session, SQLiteSession

from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)

from .closeable_session import (
    CloseableSession,
)

SDK_SESSIONS_TABLE = "agent_sdk_sessions"
SDK_MESSAGES_TABLE = "agent_sdk_messages"


@dataclass(frozen=True, slots=True)
class SQLiteSessionFactory:
    """Create local Agents SDK sessions for feature-scoped runs."""

    database_path: Path

    def create_session(self, context: AgentContextEnvelope) -> Session:
        """Create a bounded SDK session for one runtime execution."""
        return SQLiteSession(
            session_id=context.session_id,
            db_path=self.database_path,
            sessions_table=SDK_SESSIONS_TABLE,
            messages_table=SDK_MESSAGES_TABLE,
            session_settings={
                "limit": context.max_conversation_history_items,
            },
        )


def no_session(_context: AgentContextEnvelope) -> Session | None:
    """Return no runtime session."""
    return None


def close_session(session: Session | None) -> None:
    """Close a concrete SDK session when the implementation supports it."""
    if isinstance(session, CloseableSession):
        session.close()


SessionFactory = Callable[[AgentContextEnvelope], Session | None]
