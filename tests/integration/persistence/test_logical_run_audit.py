"""Integration tests for one logical audit run across model segments."""

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
