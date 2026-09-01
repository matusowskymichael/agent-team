"""Integration tests for SQLite agent session persistence."""

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
