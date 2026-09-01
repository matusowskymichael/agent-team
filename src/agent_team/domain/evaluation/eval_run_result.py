"""Evaluation run result domain model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.evaluation.eval_case_result import EvalCaseResult


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """Persisted result for a complete evaluation suite run."""

    id: str
    suite_id: str
    candidate_model: str
    judge_model: str | None
    dataset_hash: str
    rubric_hash: str
    instructions_hash: str
    package_version: str
    started_at: datetime
    ended_at: datetime
    case_results: tuple[EvalCaseResult, ...]
    warnings: tuple[str, ...]
    case_filter: str | None = None
    duration_seconds: float | None = None
    candidate_thinking_enabled: bool = False
    judge_thinking_enabled: bool | None = None
