"""Evaluation run configuration domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalRunConfig:
    """Configuration for one local evaluation suite run."""

    candidate_model: str
    instructions_hash: str
    repetitions: int = 1
    judge_model: str | None = None
    judge_repetitions: int = 1
    case_id: str | None = None
    infrastructure_retries: int = 1
    candidate_thinking_enabled: bool = False
    judge_thinking_enabled: bool | None = None
