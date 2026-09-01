"""Tests for the SQLite agent audit repository."""

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_denial import ToolInvocationDenial
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)


class TestSQLiteAgentAuditRepository:
    """SQLite audit repository behavior tests."""

    def test_successful_runs_are_recorded_and_finalized(
        self,
        tmp_path: Path,
    ) -> None:
        """Store and complete agent run records."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )

        run = _start_run(
            repository,
            _run_start(prompt_excerpt="Create a feature."),
        )
        completed = repository.complete_run(
            run_id=run.id,
            output_hash="output-hash",
            output_excerpt="Created feature 1.",
        )

        assert run.status is AgentRunStatus.STARTED
        assert completed.status is AgentRunStatus.COMPLETED
        assert completed.ended_at is not None
        assert completed.output_hash == "output-hash"
        assert completed.output_excerpt == "Created feature 1."
        assert completed.generation_metadata is None

    def test_generation_metadata_is_sanitized_and_persisted(
        self,
        tmp_path: Path,
    ) -> None:
        """Store only allowlisted generation metadata fields."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )
        run = _start_run(repository)
        metadata = AgentGenerationMetadata(
            finish_reason="<think>secret</think>length",
            input_tokens=12,
            output_tokens=34,
            visible_output_char_count=56,
            objectively_truncated=True,
            model="qwen3.6:27b",
        )

        completed = repository.complete_run(
            run_id=run.id,
            output_hash="output-hash",
            output_excerpt="Partial response.",
            generation_metadata=metadata,
        )

        assert completed.generation_metadata is not None
        assert completed.generation_metadata.finish_reason == (
            "[hidden reasoning omitted]length"
        )
        assert completed.generation_metadata.input_tokens == 12
        assert completed.generation_metadata.output_tokens == 34
        assert completed.generation_metadata.objectively_truncated is True

    def test_feature_scoped_run_metadata_is_recorded(
        self,
        tmp_path: Path,
    ) -> None:
        """Store selected model, feature ID, and session ID on the run."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )

        run = _start_run(
            repository,
            _run_start(
                model="llama3.2:3b",
                feature_id=11,
                session_id="session-11",
            ),
        )

        assert run.model == "llama3.2:3b"
        assert run.feature_id == 11
        assert run.session_id == "session-11"

    def test_failed_runs_are_recorded(
        self,
        tmp_path: Path,
    ) -> None:
        """Store sanitized failure details supplied by the harness."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )
        run = _start_run(
            repository,
            _run_start(prompt_excerpt="Create a feature."),
        )

        failed = repository.fail_run(
            run_id=run.id,
            error_type="RuntimeError",
            error_message="Local model failed.",
        )

        assert failed.status is AgentRunStatus.FAILED
        assert failed.ended_at is not None
        assert failed.error_type == "RuntimeError"
        assert failed.error_message == "Local model failed."

    def test_failed_runs_can_record_generation_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        """Store incomplete-output metadata on failed runs."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )
        run = _start_run(repository)
        metadata = AgentGenerationMetadata(
            finish_reason="length",
            input_tokens=20,
            output_tokens=8192,
            visible_output_char_count=400,
            objectively_truncated=True,
            model="qwen3.6:27b",
        )

        recorded = repository.record_run_generation_metadata(
            run_id=run.id,
            output_hash="partial-output-hash",
            output_excerpt="Partial output.",
            generation_metadata=metadata,
        )
        failed = repository.fail_run(
            run_id=run.id,
            error_type="AgentOutputIncompleteError",
            error_message="The model reached its output limit.",
        )

        assert recorded.generation_metadata == metadata
        assert failed.status is AgentRunStatus.FAILED
        assert failed.output_hash == "partial-output-hash"
        assert failed.output_excerpt == "Partial output."
        assert failed.generation_metadata == metadata

    def test_successful_tool_calls_reference_their_run(
        self,
        tmp_path: Path,
    ) -> None:
        """Store and complete tool invocations for a containing run."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )
        run = _start_run(
            repository,
            _run_start(prompt_excerpt="List features."),
        )

        invocation = repository.start_tool_invocation(
            _tool_invocation_start(run.id),
        )
        completed = repository.complete_tool_invocation(
            invocation_id=invocation.id,
            result_hash="result-hash",
            result_preview='{"result":[]}',
        )

        assert invocation.status is ToolInvocationStatus.ALLOWED
        assert completed.status is ToolInvocationStatus.COMPLETED
        assert completed.run_id == run.id
        assert completed.result_hash == "result-hash"

    def test_failed_tool_calls_are_recorded(
        self,
        tmp_path: Path,
    ) -> None:
        """Finalize failed tool invocation records."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )
        run = _start_run(
            repository,
            _run_start(prompt_excerpt="List features."),
        )
        invocation = repository.start_tool_invocation(
            _tool_invocation_start(run.id),
        )

        failed = repository.fail_tool_invocation(
            invocation_id=invocation.id,
            error_type="RuntimeError",
            error_message="Tool failed.",
        )

        assert failed.status is ToolInvocationStatus.FAILED
        assert failed.ended_at is not None
        assert failed.error_type == "RuntimeError"

    def test_denied_tool_calls_are_recorded_without_result(
        self,
        tmp_path: Path,
    ) -> None:
        """Store denied tool invocation records."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )
        run = _start_run(
            repository,
            _run_start(
                role=DevelopmentRole.BUSINESS_ANALYST,
                prompt_excerpt="Create a feature.",
            ),
        )

        denied = repository.deny_tool_invocation(
            ToolInvocationDenial(
                invocation=_tool_invocation_start(
                    run.id,
                    tool_name="create_feature",
                    classification=ToolClassification.MUTATING,
                ),
                error_type="CapabilityDeniedError",
                error_message="Tool is not allowed.",
            ),
        )

        assert denied.status is ToolInvocationStatus.DENIED
        assert denied.result_hash is None
        assert denied.ended_at is not None

    def test_missing_run_cannot_receive_tool_invocation(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject tool invocation records without a valid run."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "workflow.db"
        )

        with pytest.raises(sqlite3.IntegrityError):
            repository.start_tool_invocation(
                _tool_invocation_start(999),
            )

    def test_foreign_keys_and_indexes_exist(
        self,
        tmp_path: Path,
        sqlite_connection: Callable[[Path], sqlite3.Connection],
    ) -> None:
        """Create required relational constraints and indexes."""
        database_path = tmp_path / "workflow.db"
        audit_repository_module.SQLiteAgentAuditRepository(database_path)

        connection = sqlite_connection(database_path)
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(tool_invocations)",
        ).fetchall()
        agent_run_indexes = _index_names(connection, "agent_runs")
        invocation_indexes = _index_names(connection, "tool_invocations")

        assert any(
            row[2] == "agent_runs" and row[3] == "run_id"
            for row in foreign_keys
        )
        assert {
            "idx_agent_runs_role",
            "idx_agent_runs_status",
            "idx_agent_runs_started_at",
            "idx_agent_runs_session_id",
            "idx_agent_runs_feature_id",
        }.issubset(agent_run_indexes)
        assert {
            "idx_tool_invocations_run_id",
            "idx_tool_invocations_tool_name",
            "idx_tool_invocations_status",
        }.issubset(invocation_indexes)


def _index_names(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    rows = connection.execute(f"PRAGMA index_list({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _start_run(
    repository: audit_repository_module.SQLiteAgentAuditRepository,
    run: AgentRunStart | None = None,
) -> AgentRunRecord:
    return repository.start_run(run or _run_start())


def _run_start(
    role: DevelopmentRole = DevelopmentRole.DELIVERY_MANAGER,
    model: str = "qwen3.5:9b",
    feature_id: int | None = None,
    session_id: str | None = None,
    prompt_excerpt: str = "Create a feature.",
) -> AgentRunStart:
    return AgentRunStart(
        role=role,
        model=model,
        prompt_hash="prompt-hash",
        prompt_excerpt=prompt_excerpt,
        max_turns=6,
        session_id=session_id,
        feature_id=feature_id,
    )


def _tool_invocation_start(
    run_id: int,
    tool_name: str = "list_features",
    classification: ToolClassification = ToolClassification.READ_ONLY,
) -> ToolInvocationStart:
    return ToolInvocationStart(
        run_id=run_id,
        server_name="development_workflow",
        tool_name=tool_name,
        classification=classification,
        arguments_hash="arguments-hash",
        arguments_preview_json="{}",
    )
