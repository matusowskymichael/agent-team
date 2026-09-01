"""SQLite-backed agent session binding repository."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.sessions.agent_session_metadata import (
    AgentSessionMetadata,
)

SESSION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agent_session_bindings (
    session_id TEXT PRIMARY KEY,
    feature_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_session_bindings_feature_id
ON agent_session_bindings(feature_id);

CREATE INDEX IF NOT EXISTS idx_agent_session_bindings_role
ON agent_session_bindings(role);
"""


@dataclass(frozen=True, slots=True)
class SQLiteAgentSessionRepository:
    """SQLite implementation of local session binding persistence."""

    database_path: Path

    def __post_init__(self) -> None:
        """Create the database directory and session schema."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.executescript(SESSION_SCHEMA_SQL)

    def get_session(
        self,
        session_id: str,
    ) -> AgentSessionMetadata | None:
        """Return stored session metadata, if it exists."""
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT
                    session_id,
                    feature_id,
                    role,
                    created_at,
                    updated_at
                FROM agent_session_bindings
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            return None if row is None else _map_session(row)

    def create_session(
        self,
        session_id: str,
        feature_id: int,
        role: DevelopmentRole,
    ) -> AgentSessionMetadata:
        """Persist a new local session binding."""
        timestamp = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO agent_session_bindings (
                    session_id,
                    feature_id,
                    role,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    feature_id,
                    role.value,
                    timestamp,
                    timestamp,
                ),
            )
            return _select_session(connection, session_id)

    def touch_session(self, session_id: str) -> AgentSessionMetadata:
        """Update a session timestamp and return its metadata."""
        timestamp = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE agent_session_bindings
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (timestamp, session_id),
            )
            return _select_session(connection, session_id)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _select_session(
    connection: sqlite3.Connection,
    session_id: str,
) -> AgentSessionMetadata:
    row = connection.execute(
        """
        SELECT
            session_id,
            feature_id,
            role,
            created_at,
            updated_at
        FROM agent_session_bindings
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("SQLite did not return the agent session.")
    return _map_session(row)


def _map_session(row: sqlite3.Row) -> AgentSessionMetadata:
    return AgentSessionMetadata(
        session_id=str(row["session_id"]),
        feature_id=int(row["feature_id"]),
        role=DevelopmentRole(str(row["role"])),
        created_at=_parse_timestamp(str(row["created_at"])),
        updated_at=_parse_timestamp(str(row["updated_at"])),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)
