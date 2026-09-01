"""Judge grade domain model."""

from dataclasses import dataclass, field

from agent_team.domain.evaluation.eval_verdict import EvalVerdict


def _empty_evidence() -> dict[str, str]:
    return {}


@dataclass(frozen=True, slots=True)
class JudgeGrade:
    """Validated result from a local rubric judge."""

    verdict: EvalVerdict
    scores: dict[str, int]
    reasons: dict[str, str]
    confidence: float
    ambiguous: bool
    error_message: str | None = None
    rubric_id: str | None = None
    rubric_version: str | None = None
    case_id: str | None = None
    evidence: dict[str, str] = field(default_factory=_empty_evidence)
    judge_model: str | None = None
    response_hash: str | None = None
    response_preview: str | None = None
    validation_errors: tuple[str, ...] = ()
    retry_count: int = 0
    raw_response: str | None = field(
        default=None,
        compare=False,
        repr=False,
    )
