"""Evaluation case result domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.deterministic_grade import DeterministicGrade
from agent_team.domain.evaluation.eval_case_intent import EvalCaseIntent
from agent_team.domain.evaluation.eval_context_policy import (
    EvalContextPolicy,
)
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.judge_grade import JudgeGrade


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """Result for one case repetition."""

    case_id: str
    repetition: int
    candidate_result: CandidateRunResult
    deterministic_grade: DeterministicGrade
    judge_grade: JudgeGrade | None
    verdict: EvalVerdict
    semantic_judge_required: bool = True
    intent: EvalCaseIntent = EvalCaseIntent.UNSPECIFIED
    context_policy: EvalContextPolicy = (
        EvalContextPolicy.STANDARD_FEATURE_CONTEXT
    )
    candidate_duration_seconds: float | None = None
    deterministic_duration_seconds: float | None = None
    judge_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
