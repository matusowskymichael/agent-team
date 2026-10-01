"""Integration tests for one logical audit run across model segments."""

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)


class TestLogicalRunAudit:
    """Persist progress independently of final logical run completion."""

    @pytest.mark.parametrize(
        "metadata",
        [
            [],
            {"model": 123},
            {"model": "local", "input_tokens": "invalid"},
            {"model": "local", "visible_output_char_count": "invalid"},
            {
                "model": "local",
                "visible_output_char_count": 0,
                "objectively_truncated": "invalid",
            },
        ],
    )
    def test_corrupt_generation_metadata(
        self,
        legacy_audit_database: Path,
        sqlite_connection: Callable[[Path], sqlite3.Connection],
        metadata: dict[str, object] | list[object],
    ) -> None:
        """Reject malformed persisted metadata instead of inventing values."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            legacy_audit_database,
        )
        payload: object = (
            {
                "model": "local",
                "visible_output_char_count": 0,
                "objectively_truncated": False,
                **metadata,
            }
            if isinstance(metadata, dict)
            else metadata
        )
        connection = sqlite_connection(legacy_audit_database)
        connection.execute(
            "UPDATE agent_runs SET generation_metadata_json = ? WHERE id = 1",
            (json.dumps(payload),),
        )
        connection.commit()

        with pytest.raises(RuntimeError, match="invalid generation metadata"):
            repository.get_run(1)

    def test_metadata_omits_optional_values(
        self,
        legacy_audit_database: Path,
        sqlite_connection: Callable[[Path], sqlite3.Connection],
    ) -> None:
        """Read older valid metadata without optional provider fields."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            legacy_audit_database,
        )
        connection = sqlite_connection(legacy_audit_database)
        connection.execute(
            "UPDATE agent_runs SET generation_metadata_json = ? WHERE id = 1",
            (
                json.dumps(
                    {
                        "model": "qwen3.5:9b",
                        "visible_output_char_count": 0,
                        "objectively_truncated": False,
                    }
                ),
            ),
        )
        connection.commit()

        run = repository.get_run(1)

        assert run is not None
        assert run.generation_metadata is not None
        assert run.generation_metadata.finish_reason is None
        assert run.generation_metadata.input_tokens is None
        assert run.generation_metadata.output_tokens is None

    def test_unknown_finalization_does_not_create_run(
        self,
        legacy_audit_database: Path,
    ) -> None:
        """Roll back failed finalization without creating an unrelated run."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            legacy_audit_database,
        )
        before = repository.list_runs(10)

        with pytest.raises(RuntimeError, match="did not return the agent run"):
            repository.record_run_progress(404, 1, "completed")

        assert repository.list_runs(10) == before

    @pytest.mark.parametrize("limit", [None, 25])
    def test_logical_run_progress_survives_reopening_and_completion(
        self,
        tmp_path: Path,
        limit: int | None,
    ) -> None:
        """Count segments on one row while preserving original binding."""
        database_path = tmp_path / "audit.db"
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            database_path,
        )
        started = repository.start_run(
            AgentRunStart(
                role=DevelopmentRole.BACKEND_DEVELOPER,
                model="qwen3.5:9b",
                prompt_hash="original-prompt-hash",
                prompt_excerpt="Implement the assigned task.",
                max_turns=10,
                total_turn_limit=limit,
                session_id="trusted-session",
                feature_id=2,
                task_id=3,
                workspace_identity_hash="trusted-workspace",
            ),
        )
        assert started.segment_count == 0
        assert started.total_turn_limit == limit

        for segment_count in range(1, 5):
            progress = repository.record_run_progress(
                started.id,
                segment_count,
            )
            assert progress.status is AgentRunStatus.STARTED
            assert progress.segment_count == segment_count
            assert progress.termination_reason is None

        repository.record_run_progress(started.id, 4, "task_completed")
        completed = repository.complete_run(
            started.id,
            "final-output-hash",
            "Task verified.",
        )
        reopened = audit_repository_module.SQLiteAgentAuditRepository(
            database_path,
        )
        assert reopened.get_run(started.id) == completed
        assert reopened.list_runs(limit=10) == [completed]
        assert completed.segment_count == 4
        assert completed.termination_reason == "task_completed"
        assert completed.total_turn_limit == limit
        assert completed.max_turns == 10
        assert completed.prompt_hash == started.prompt_hash
        assert completed.started_at == started.started_at
        assert completed.session_id == started.session_id
        assert completed.feature_id == started.feature_id
        assert completed.task_id == started.task_id
        assert completed.workspace_identity_hash == (
            started.workspace_identity_hash
        )

    def test_progress_termination_survives_failure(
        self,
        tmp_path: Path,
    ) -> None:
        """Keep stall evidence when the logical run fails."""
        repository = audit_repository_module.SQLiteAgentAuditRepository(
            tmp_path / "audit.db",
        )
        started = repository.start_run(
            AgentRunStart(
                role=DevelopmentRole.BACKEND_DEVELOPER,
                model="qwen3.5:9b",
                prompt_hash="original-prompt-hash",
                prompt_excerpt="Implement the assigned task.",
                max_turns=10,
            ),
        )

        repository.record_run_progress(started.id, 5, "stalled")
        failed = repository.fail_run(
            started.id,
            "AgentStalledError",
            "Repeated segments produced no new progress.",
        )

        assert failed.status is AgentRunStatus.FAILED
        assert failed.segment_count == 5
        assert failed.termination_reason == "stalled"
        assert failed.error_type == "AgentStalledError"
