"""Human evaluation label domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.eval_verdict import EvalVerdict


@dataclass(frozen=True, slots=True)
class HumanLabel:
    """Human rubric label used for judge calibration."""

    case_id: str
    rubric_id: str
    rubric_version: str
    scores: dict[str, int]
    verdict: EvalVerdict
    reason: str
    rater: str
    rated_at: str
