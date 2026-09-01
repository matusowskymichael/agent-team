"""Tests for the human audit CLI."""

from dataclasses import replace

import pytest
from pytest import CaptureFixture

from agent_team.application.audit.audit_query_service import AuditQueryService
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_audit_migration_error,
)
from agent_team.interfaces.cli import audit_cli
from tests.unit.fakes.audit.audit_record_factories import (
    make_agent_run_record,
    make_tool_invocation_record,
)
from tests.unit.fakes.audit.fake_agent_audit_reader import FakeAgentAuditReader


class TestAuditCli:
    """Human audit CLI behavior tests."""

    def test_list_runs_command_displays_sanitized_run_fields(
        self,
        capsys: CaptureFixture[str],
    ) -> None:
        """Print run rows from already-sanitized audit records."""
        reader = FakeAgentAuditReader(
            runs=[
                make_agent_run_record(run_id=2),
                make_agent_run_record(run_id=1),
            ],
        )
        service = AuditQueryService(reader=reader)

        exit_code = audit_cli.main(
            ["list-runs", "--limit", "1"],
            service=service,
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert reader.received_limits == [1]
        assert "ID\tSTATUS\tROLE\tMODEL\tSTARTED\tPROMPT" in captured.out
        assert "2\tcompleted\tdelivery_manager" in captured.out
        assert "Create a login feature." in captured.out
        assert "1\tcompleted" not in captured.out
        assert captured.err == ""

    def test_show_run_command_displays_invocations(
        self,
        capsys: CaptureFixture[str],
    ) -> None:
        """Print one run and its associated tool invocations."""
        reader = FakeAgentAuditReader(
            runs=[make_agent_run_record(run_id=1)],
            tool_invocations=[make_tool_invocation_record(run_id=1)],
        )
        service = AuditQueryService(reader=reader)

        exit_code = audit_cli.main(["show-run", "1"], service=service)

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Run 1" in captured.out
        assert "Feature ID: -" in captured.out
        assert "Session ID: -" in captured.out
        assert "Prompt: Create a login feature." in captured.out
        assert "Output: Created feature 1." in captured.out
        assert "Generation metadata: -" in captured.out
        assert "Tool invocations:" in captured.out
        assert "development_workflow.create_feature" in captured.out
        assert 'Arguments: {"title":"Login"}' in captured.out
        assert 'Result: {"id":1}' in captured.out
        assert captured.err == ""

    def test_show_run_command_displays_skill_load_hash(
        self,
        capsys: CaptureFixture[str],
    ) -> None:
        """Show skill loads as auxiliary audited tool invocations."""
        skill_invocation = replace(
            make_tool_invocation_record(run_id=1),
            server_name="agent_skills",
            tool_name="load_skill",
            classification=ToolClassification.READ_ONLY,
            arguments_preview_json='{"name":"write-requirements-artifact"}',
            result_preview=(
                '{"content_hash":"skill-hash","loaded":true,'
                '"name":"write-requirements-artifact"}'
            ),
        )
        reader = FakeAgentAuditReader(
            runs=[make_agent_run_record(run_id=1)],
            tool_invocations=[skill_invocation],
        )
        service = AuditQueryService(reader=reader)

        exit_code = audit_cli.main(["show-run", "1"], service=service)

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "agent_skills.load_skill" in captured.out
        assert "write-requirements-artifact" in captured.out
        assert "skill-hash" in captured.out

    def test_show_run_command_displays_generation_metadata(
        self,
        capsys: CaptureFixture[str],
    ) -> None:
        """Print sanitized generation metadata for human audit."""
        metadata = AgentGenerationMetadata(
            finish_reason="length",
            input_tokens=10,
            output_tokens=8192,
            visible_output_char_count=1200,
            objectively_truncated=True,
            model="qwen3.6:27b",
        )
        reader = FakeAgentAuditReader(
            runs=[
                make_agent_run_record(
                    run_id=1,
                    generation_metadata=metadata,
                ),
            ],
        )
        service = AuditQueryService(reader=reader)

        exit_code = audit_cli.main(["show-run", "1"], service=service)

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Generation metadata:" in captured.out
        assert "Finish reason: length" in captured.out
        assert "Input tokens: 10" in captured.out
        assert "Output tokens: 8192" in captured.out
        assert "Visible output chars: 1200" in captured.out
        assert "Objectively truncated: True" in captured.out
        assert "Model: qwen3.6:27b" in captured.out

    def test_show_run_unknown_id_returns_concise_error(
        self,
        capsys: CaptureFixture[str],
    ) -> None:
        """Return non-zero with a concise error for missing runs."""
        service = AuditQueryService(reader=FakeAgentAuditReader())

        exit_code = audit_cli.main(["show-run", "404"], service=service)

        captured = capsys.readouterr()
        assert exit_code == 1
        assert captured.out == ""
        assert captured.err == "Agent run 404 was not found.\n"

    def test_database_migration_failure_returns_concise_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: CaptureFixture[str],
    ) -> None:
        """Return non-zero without a traceback for migration failures."""

        def build_audit_query_service() -> AuditQueryService:
            raise sqlite_audit_migration_error.SQLiteAuditMigrationError(
                "Audit database migration failed."
            )

        monkeypatch.setattr(
            audit_cli,
            "build_audit_query_service",
            build_audit_query_service,
        )

        exit_code = audit_cli.main(["list-runs"])

        captured = capsys.readouterr()
        assert exit_code == 1
        assert captured.out == ""
        assert captured.err == (
            "Database migration failed: Audit database migration failed.\n"
        )
