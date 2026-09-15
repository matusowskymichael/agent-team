"""Integration tests for SQLite agent session persistence."""

import sqlite3
from contextlib import closing
from pathlib import Path

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_agent_session_repository as session_repository_module,
)


class TestSQLiteAgentSessionRepository:
    """SQLiteAgentSessionRepository behavior tests."""

    def test_creates_and_touches_session_binding(
        self,
        tmp_path: Path,
    ) -> None:
        """Persist role-and-feature session metadata locally."""
        repository = session_repository_module.SQLiteAgentSessionRepository(
            tmp_path / "workflow.db"
        )

        created = repository.create_session(
            session_id="feature-1",
            feature_id=1,
            role=DevelopmentRole.BUSINESS_ANALYST,
        )
        touched = repository.touch_session("feature-1")
        loaded = repository.get_session("feature-1")

        assert loaded == touched
        assert created.session_id == "feature-1"
        assert touched.feature_id == 1
        assert touched.role is DevelopmentRole.BUSINESS_ANALYST
        assert touched.updated_at >= created.updated_at

    def test_migrates_legacy_session_rows_with_null_task_scope(
        self,
        tmp_path: Path,
    ) -> None:
        """Keep historical sessions readable after task-scope migration."""
        database_path = tmp_path / "workflow.db"
        with closing(sqlite3.connect(database_path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE agent_session_bindings (
                    session_id TEXT PRIMARY KEY,
                    feature_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
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
                    "legacy-backend",
                    1,
                    "backend_developer",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )

        repository = session_repository_module.SQLiteAgentSessionRepository(
            database_path,
        )
        legacy = repository.get_session("legacy-backend")
        created = repository.create_session(
            session_id="backend-task",
            feature_id=1,
            role=DevelopmentRole.BACKEND_DEVELOPER,
            task_id=2,
            workspace_identity_hash="workspace-hash",
        )

        assert legacy is not None
        assert legacy.task_id is None
        assert legacy.workspace_identity_hash is None
        assert created.task_id == 2
        assert created.workspace_identity_hash == "workspace-hash"
        assert {
            "task_id",
            "workspace_identity_hash",
        }.issubset(_columns(database_path))
        assert {
            "idx_agent_session_bindings_task_id",
            "idx_agent_session_bindings_workspace_identity",
        }.issubset(_indexes(database_path))


def _columns(database_path: Path) -> set[str]:
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            "PRAGMA table_info(agent_session_bindings)",
        ).fetchall()
    return {str(row[1]) for row in rows}


def _indexes(database_path: Path) -> set[str]:
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            "PRAGMA index_list(agent_session_bindings)",
        ).fetchall()
    return {str(row[1]) for row in rows}
