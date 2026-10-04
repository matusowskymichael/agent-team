"""Atomic checkpoint persistence and compatibility for interrupted runs."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_team.domain.evaluation.eval_run_status import EvalRunStatus
from agent_team.domain.runtime.agent_lifecycle_phase import AgentLifecyclePhase
from agent_team.domain.runtime.agent_liveness_snapshot import (
    AgentLivenessSnapshot,
)
from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)
from agent_team.infrastructure.evaluation import json_eval_result_repository
from agent_team.infrastructure.evaluation.json_eval_result_repository import (
    JsonEvalResultRepository,
)


class TestAtomicEvalCheckpoint:
    """Leave valid old JSON in place if a checkpoint is interrupted."""

    def test_atomic_checkpoint_survives_interrupted_replacement(
        self,
        evaluation_result_record: dict[str, object],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Preserve prior diagnostics and remove the staged partial file."""
        assert evaluation_result_record["id"] == "boundary-run"
        repository = JsonEvalResultRepository(tmp_path)
        original = repository.get("boundary-run")
        assert original is not None

        def cancel_replace(_source: Path, _destination: Path) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(
            json_eval_result_repository.os,
            "replace",
            cancel_replace,
        )
        with pytest.raises(KeyboardInterrupt):
            repository.save(
                replace(original, status=EvalRunStatus.INTERRUPTED)
            )

        assert repository.get("boundary-run") == original
        assert sorted(item.name for item in tmp_path.iterdir()) == [
            "boundary-run.json",
        ]

    @pytest.mark.parametrize("status", list(EvalRunStatus))
    def test_checkpoint_round_trip_preserves_diagnostics(
        self,
        evaluation_result_record: dict[str, object],
        tmp_path: Path,
        status: EvalRunStatus,
    ) -> None:
        """Preserve safe liveness facts under every durable run status."""
        assert evaluation_result_record["id"] == "boundary-run"
        repository = JsonEvalResultRepository(tmp_path)
        original = repository.get("boundary-run")
        assert original is not None
        candidate = replace(
            original.case_results[0].candidate_result,
            liveness_snapshot=AgentLivenessSnapshot(
                segment_count=3,
                turns_used=23,
                task_status="in_progress",
                lifecycle_phase=AgentLifecyclePhase.CHECKED,
                last_tool_name="run_check",
                changed_paths=("backend/auth.py",),
                last_check_outcome="failed",
                last_verification_outcome="failed",
                verification_failure_classification="check_failed",
                time_since_advancement_seconds=900,
                elapsed_seconds=1800,
                waiting_phase="cleanup",
            ),
            termination_reason="case_timeout",
        )
        checkpoint = replace(
            original,
            status=status,
            active_case_id="bd-dev-002",
            active_repetition=1,
            active_attempt=2,
            active_candidate=candidate,
            termination_type="EvalCaseTimeoutError",
            termination_message="Candidate safety deadline reached.",
            case_timeout_seconds=2700,
            runtime_watchdogs=AgentWatchdogSettings(),
        )

        repository.save(checkpoint)

        assert repository.get(checkpoint.id) == checkpoint
        assert repository.list_ids() == ["boundary-run"]

    def test_historical_run_status_defaults_to_completed(
        self,
        evaluation_result_record: dict[str, object],
        tmp_path: Path,
    ) -> None:
        """Read old result files that predate safety diagnostics and status."""
        for key in (
            "status",
            "active_case_id",
            "active_repetition",
            "active_attempt",
            "active_candidate",
            "termination_type",
            "termination_message",
            "case_timeout_seconds",
            "runtime_watchdogs",
        ):
            evaluation_result_record.pop(key, None)
        (tmp_path / "boundary-run.json").write_text(
            json.dumps(evaluation_result_record),
            encoding="utf-8",
        )

        result = JsonEvalResultRepository(tmp_path).get("boundary-run")

        assert result is not None
        assert result.status is EvalRunStatus.COMPLETED
        assert result.active_candidate is None
        assert result.case_timeout_seconds is None
        assert (
            result.case_results[0].candidate_result.liveness_snapshot is None
        )
