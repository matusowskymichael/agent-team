"""Evaluation progress event kinds."""

from enum import StrEnum


class EvalProgressEventKind(StrEnum):
    """Progress lifecycle event names for evaluation runs."""

    RUN_STARTED = "run_started"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    INFRASTRUCTURE_RETRY = "infrastructure_retry"
    CASE_COMPLETED = "case_completed"
    RUN_FINISHED = "run_finished"
    RUN_CANCELLED = "run_cancelled"
