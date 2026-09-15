"""Persistence integration test fixtures."""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path

import pytest

from tests.reporting.allure_steps import fixture_title


@pytest.fixture
@fixture_title("Create a legacy audit database")
def legacy_audit_database(tmp_path: Path) -> Path:
    """Create a legacy audit database and close setup resources."""
    database_path = tmp_path / "workflow.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        _create_legacy_audit_schema(connection)
        _insert_legacy_audit_rows(connection)
    return database_path


@pytest.fixture
@fixture_title("Open managed SQLite integration connections")
def sqlite_connection() -> Iterator[Callable[[Path], sqlite3.Connection]]:
    """Yield SQLite connections and always close them after tests."""
    connections: list[sqlite3.Connection] = []

    def connect(database_path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(database_path)
        connections.append(connection)
        return connection

    yield connect

    for connection in connections:
        connection.close()


def _create_legacy_audit_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            prompt_excerpt TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            max_turns INTEGER NOT NULL,
            output_hash TEXT,
            output_excerpt TEXT,
            error_type TEXT,
            error_message TEXT
        )
        """,
    )
    connection.execute(
        """
        CREATE TABLE tool_invocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            server_name TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            classification TEXT NOT NULL,
            status TEXT NOT NULL,
            arguments_hash TEXT NOT NULL,
            arguments_preview_json TEXT NOT NULL,
            result_hash TEXT,
            result_preview TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            error_type TEXT,
            error_message TEXT,
            FOREIGN KEY (run_id) REFERENCES agent_runs(id)
                ON DELETE CASCADE
        )
        """,
    )
    for index_sql in _LEGACY_INDEX_SQL:
        connection.execute(index_sql)


def _insert_legacy_audit_rows(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO agent_runs (
            id,
            role,
            model,
            status,
            prompt_hash,
            prompt_excerpt,
            started_at,
            ended_at,
            max_turns,
            output_hash,
            output_excerpt,
            error_type,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "business_analyst",
            "qwen3.5:9b",
            "completed",
            "legacy-prompt-hash",
            "List features.",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:02+00:00",
            6,
            "legacy-output-hash",
            "Feature list returned.",
            None,
            None,
        ),
    )
    connection.execute(
        """
        INSERT INTO tool_invocations (
            id,
            run_id,
            server_name,
            tool_name,
            classification,
            status,
            arguments_hash,
            arguments_preview_json,
            result_hash,
            result_preview,
            started_at,
            ended_at,
            error_type,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            1,
            "development_workflow",
            "list_features",
            "read_only",
            "completed",
            "legacy-arguments-hash",
            "{}",
            "legacy-result-hash",
            "[]",
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:02+00:00",
            None,
            None,
        ),
    )


_LEGACY_INDEX_SQL = (
    """
    CREATE INDEX idx_agent_runs_role
    ON agent_runs(role)
    """,
    """
    CREATE INDEX idx_agent_runs_status
    ON agent_runs(status)
    """,
    """
    CREATE INDEX idx_agent_runs_started_at
    ON agent_runs(started_at)
    """,
    """
    CREATE INDEX idx_tool_invocations_run_id
    ON tool_invocations(run_id)
    """,
    """
    CREATE INDEX idx_tool_invocations_tool_name
    ON tool_invocations(tool_name)
    """,
    """
    CREATE INDEX idx_tool_invocations_status
    ON tool_invocations(status)
    """,
)
