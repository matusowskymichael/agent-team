"""Terminal lifecycle fixtures."""

from pathlib import Path

import pytest

from agent_team.application.runtime.orchestrator import Orchestrator
from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_event import EvalProgressEvent
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)
from agent_team.infrastructure.configuration.workflow_database_path import (
    AGENT_TEAM_DB_PATH_ENV,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.interfaces.cli.agent_cli import build_orchestrator


@pytest.fixture
def local_runtime_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Orchestrator:
    """Compose local persistence and cancellation without model execution."""
    monkeypatch.setenv(AGENT_TEAM_DB_PATH_ENV, str(tmp_path / "workflow.db"))
    return build_orchestrator(OllamaSettings())


@pytest.fixture
def terminal_progress_event() -> EvalProgressEvent:
    """Provide an active evaluation event with no model or workspace data."""
    return EvalProgressEvent(
        kind=EvalProgressEventKind.RUN_STARTED,
        suite_id="frontend_developer_development",
        completed_cases=0,
        total_cases=1,
        elapsed_seconds=0,
        case_id="fd-dev-002",
        phase=EvalPhase.CANDIDATE,
        repetition=1,
    )
