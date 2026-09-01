"""Evaluation candidate attempt result domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalAttemptResult:
    """Minimal persisted metadata for one candidate execution attempt."""

    attempt: int
    status: str
    duration_seconds: float | None = None
    error_type: str | None = None
    error_stage: str | None = None
