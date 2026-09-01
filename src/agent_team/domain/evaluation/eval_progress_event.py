"""Evaluation progress event domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)


@dataclass(frozen=True, slots=True)
class EvalProgressEvent:
    """Observable progress state for an evaluation run."""

    kind: EvalProgressEventKind
    suite_id: str
    completed_cases: int
    total_cases: int
    elapsed_seconds: float
    case_id: str | None = None
    phase: EvalPhase | None = None
    repetition: int | None = None
    total_repetitions: int = 1
    judge_repetition: int | None = None
    total_judge_repetitions: int | None = None
    estimated_remaining_seconds: float | None = None
    phase_duration_seconds: float | None = None
    case_duration_seconds: float | None = None
    infrastructure_retry: int | None = None
    total_infrastructure_retries: int | None = None
