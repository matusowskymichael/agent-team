"""Isolated CLI composition fixtures without live inference."""

import pytest

from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_event import EvalProgressEvent
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)
from agent_team.domain.runtime.agent_lifecycle_phase import AgentLifecyclePhase
from agent_team.domain.runtime.agent_liveness_snapshot import (
    AgentLivenessSnapshot,
)
from agent_team.interfaces.cli import eval_cli
from tests.unit.interfaces.cli.test_eval_cli import (
    EvalCliRepository,
    EvalCliRunner,
)


@pytest.fixture
def fake_eval_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> EvalCliRepository:
    """Reuse CLI fakes while disabling model-readiness network requests."""
    repository = EvalCliRepository()

    def ensure_model_ready(_settings: object) -> None:
        """Skip provider requests at the composition boundary."""
        return

    monkeypatch.setattr(
        eval_cli, "ensure_ollama_model_ready", ensure_model_ready
    )
    monkeypatch.setattr(eval_cli, "EvalRunner", EvalCliRunner)
    monkeypatch.setattr(
        eval_cli, "JsonEvalResultRepository", lambda: repository
    )
    return repository


@pytest.fixture
def liveness_heartbeat() -> EvalProgressEvent:
    """Supply a trusted active-candidate heartbeat without private contents."""
    return EvalProgressEvent(
        kind=EvalProgressEventKind.HEARTBEAT,
        suite_id="frontend_developer_development",
        completed_cases=0,
        total_cases=1,
        elapsed_seconds=750,
        case_id="fd-dev-002",
        phase=EvalPhase.CANDIDATE,
        repetition=1,
        total_repetitions=1,
        case_timeout_seconds=2700,
        case_remaining_seconds=1950,
        liveness_snapshot=AgentLivenessSnapshot(
            segment_count=2,
            turns_used=12,
            task_status="in_progress",
            lifecycle_phase=AgentLifecyclePhase.CHECKED,
            last_tool_name="run_check",
            last_check_outcome="failed",
            time_since_advancement_seconds=250,
            waiting_phase="model",
        ),
    )
