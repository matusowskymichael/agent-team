"""Tests for workflow database path configuration."""

from pathlib import Path

import pytest

from agent_team.infrastructure.configuration.workflow_database_path import (
    AGENT_TEAM_DB_PATH_ENV,
    DEFAULT_WORKFLOW_DB_PATH,
    load_workflow_database_path,
)


class TestWorkflowDatabasePath:
    """Workflow database path configuration tests."""

    def test_load_workflow_database_path_uses_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use the default workflow database path."""
        monkeypatch.delenv(AGENT_TEAM_DB_PATH_ENV, raising=False)

        assert load_workflow_database_path() == DEFAULT_WORKFLOW_DB_PATH

    def test_load_workflow_database_path_uses_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Use the configured workflow database path."""
        database_path = tmp_path / "workflow.db"
        monkeypatch.setenv(AGENT_TEAM_DB_PATH_ENV, str(database_path))

        assert load_workflow_database_path() == database_path

    def test_load_workflow_database_path_accepts_mapping(
        self,
        tmp_path: Path,
    ) -> None:
        """Use an explicit environment mapping."""
        database_path = tmp_path / "mapping.db"

        assert (
            load_workflow_database_path(
                {AGENT_TEAM_DB_PATH_ENV: str(database_path)},
            )
            == database_path
        )
