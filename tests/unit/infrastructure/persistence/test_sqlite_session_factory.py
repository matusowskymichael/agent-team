"""Tests for the Agents SDK SQLite session factory."""

from pathlib import Path
from typing import cast

import pytest
from agents.memory import Session

from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_session_factory,
)


class TestSQLiteSessionFactory:
    """SQLiteSessionFactory behavior tests."""

    def test_creates_bounded_sdk_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass session ID, tables, and history limit to the SDK."""
        calls: list[dict[str, object]] = []
        fake_session = cast("Session", object())
        database_path = tmp_path / "workflow.db"

        def sqlite_session(**kwargs: object) -> Session:
            calls.append(kwargs)
            return fake_session

        monkeypatch.setattr(
            sqlite_session_factory,
            "SQLiteSession",
            sqlite_session,
        )
        context = AgentContextEnvelope(
            feature_id=1,
            session_id="session-1",
            authoritative_context="context",
            max_conversation_history_items=7,
        )

        session = sqlite_session_factory.SQLiteSessionFactory(
            database_path
        ).create_session(context)

        assert session is fake_session
        assert calls == [
            {
                "session_id": "session-1",
                "db_path": database_path,
                "sessions_table": sqlite_session_factory.SDK_SESSIONS_TABLE,
                "messages_table": sqlite_session_factory.SDK_MESSAGES_TABLE,
                "session_settings": {"limit": 7},
            },
        ]
