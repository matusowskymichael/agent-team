"""Integration tests for the human audit CLI."""

from pathlib import Path

import pytest
from pytest import CaptureFixture

from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.configuration.workflow_database_path import (
    AGENT_TEAM_DB_PATH_ENV,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.interfaces.cli import audit_cli


class TestAuditCliIntegration:
    """Audit CLI integration tests using SQLite."""

    def test_list_runs_reads_configured_sqlite_database(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: CaptureFixture[str],
    ) -> None:
        """List audit runs from AGENT_TEAM_DB_PATH."""
        database_path = tmp_path / "workflow.db"
        _seed_completed_run(database_path)
        monkeypatch.setenv(AGENT_TEAM_DB_PATH_ENV, str(database_path))

        exit_code = audit_cli.main(["list-runs", "--limit", "10"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "1\tcompleted\tdelivery_manager" in captured.out
        assert "Create a feature." in captured.out
        assert captured.err == ""

    def test_show_run_reads_invocations_from_configured_sqlite_database(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: CaptureFixture[str],
    ) -> None:
        """Show one audit run and its tool calls from AGENT_TEAM_DB_PATH."""
        database_path = tmp_path / "workflow.db"
        run_id = _seed_completed_run(database_path)
        monkeypatch.setenv(AGENT_TEAM_DB_PATH_ENV, str(database_path))

        exit_code = audit_cli.main(["show-run", str(run_id)])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Run 1" in captured.out
        assert "Status: completed" in captured.out
        assert "Feature ID: 1" in captured.out
        assert "Session ID: session-1" in captured.out
        assert "Prompt: Create a feature." in captured.out
        assert "development_workflow.create_feature" in captured.out
        assert 'Arguments: {"title":"Login"}' in captured.out
        assert 'Result: {"id":1}' in captured.out
        assert captured.err == ""


def _seed_completed_run(database_path: Path) -> int:
    repository = audit_repository_module.SQLiteAgentAuditRepository(
        database_path
    )
    run = repository.start_run(
        AgentRunStart(
            role=DevelopmentRole.DELIVERY_MANAGER,
            model="qwen3.5:9b",
            prompt_hash="prompt-hash",
            prompt_excerpt="Create a feature.",
            max_turns=6,
            session_id="session-1",
            feature_id=1,
        ),
    )
    invocation = repository.start_tool_invocation(
        ToolInvocationStart(
            run_id=run.id,
            server_name="development_workflow",
            tool_name="create_feature",
            classification=ToolClassification.MUTATING,
            arguments_hash="arguments-hash",
            arguments_preview_json='{"title":"Login"}',
        ),
    )
    repository.complete_tool_invocation(
        invocation_id=invocation.id,
        result_hash="result-hash",
        result_preview='{"id":1}',
    )
    completed = repository.complete_run(
        run_id=run.id,
        output_hash="output-hash",
        output_excerpt="Created feature 1.",
    )
    return completed.id
