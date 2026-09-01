"""Application service for local rubric judging."""

from collections.abc import Mapping
from dataclasses import dataclass

from agent_team.application.audit.audit_sanitizer import sanitize_error
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_judge import EvalJudge
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.judge_correction_request import (
    JudgeCorrectionRequest,
)
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.rubric import Rubric

MAX_RUBRIC_SCORE = 4
MAX_JUDGE_REASON_LENGTH = 240


@dataclass(frozen=True, slots=True)
class RubricJudgeService:
    """Validate local judge output against a strict rubric schema."""

    judge: EvalJudge

    async def judge_case(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
        judge_repetitions: int = 1,
    ) -> JudgeGrade:
        """Judge a case, retrying invalid structured output once."""
        if judge_repetitions < 1:
            raise ValueError("judge_repetitions must be greater than zero.")

        try:
            grade = await self.judge.grade(
                case,
                rubric,
                candidate,
                judge_model,
            )
        except Exception as error:
            return _exception_grade(error, judge_model)

        errors = _validate_grade(grade, rubric, case)
        if not errors:
            return grade

        invalid_response = grade.raw_response or grade.response_preview or ""
        try:
            corrected = await self.judge.correct_grade(
                case,
                rubric,
                candidate,
                judge_model,
                JudgeCorrectionRequest(
                    invalid_response=invalid_response,
                    validation_errors=errors,
                ),
            )
        except Exception as error:
            error_type, error_message = sanitize_error(error)
            return JudgeGrade(
                verdict=EvalVerdict.JUDGE_ERROR,
                scores={},
                reasons={},
                confidence=0.0,
                ambiguous=True,
                error_message=f"{error_type}: {error_message}",
                judge_model=judge_model,
                validation_errors=errors,
                retry_count=1,
            )

        correction_errors = _validate_grade(corrected, rubric, case)
        if not correction_errors:
            return _with_retry_metadata(corrected, errors)

        return JudgeGrade(
            verdict=EvalVerdict.JUDGE_ERROR,
            scores={},
            reasons={},
            confidence=0.0,
            ambiguous=True,
            error_message="judge output remained invalid after correction",
            judge_model=judge_model,
            response_hash=corrected.response_hash,
            response_preview=corrected.response_preview,
            validation_errors=errors + correction_errors,
            retry_count=1,
        )


def _validate_grade(
    grade: JudgeGrade,
    rubric: Rubric,
    case: EvalCase,
) -> tuple[str, ...]:
    errors: list[str] = []
    if grade.verdict is EvalVerdict.JUDGE_ERROR:
        if grade.validation_errors:
            errors.extend(grade.validation_errors)
        else:
            errors.append(
                grade.error_message or "verdict: judge returned error",
            )
        return tuple(errors)
    dimension_ids = {dimension.id for dimension in rubric.dimensions}
    _validate_identity(grade, rubric, case, errors)
    _validate_scores(grade, dimension_ids, errors)
    _validate_reasons(grade, dimension_ids, errors)
    _validate_evidence(grade, dimension_ids, errors)
    _validate_verdict(grade, errors)
    return tuple(errors)


def _validate_identity(
    grade: JudgeGrade,
    rubric: Rubric,
    case: EvalCase,
    errors: list[str],
) -> None:
    if grade.rubric_id != rubric.id:
        errors.append("rubric_id: expected canonical rubric ID")
    if grade.rubric_version != rubric.version:
        errors.append("rubric_version: expected canonical rubric version")
    if grade.case_id != case.id:
        errors.append("case_id: expected selected case ID")


def _validate_scores(
    grade: JudgeGrade,
    dimension_ids: set[str],
    errors: list[str],
) -> None:
    _validate_dimension_keys("scores", grade.scores, dimension_ids, errors)
    for dimension_id, score in grade.scores.items():
        if type(score) is not int:
            errors.append(f"scores.{dimension_id}: expected integer")
        elif not 0 <= score <= MAX_RUBRIC_SCORE:
            errors.append(f"scores.{dimension_id}: expected 0 through 4")


def _validate_reasons(
    grade: JudgeGrade,
    dimension_ids: set[str],
    errors: list[str],
) -> None:
    _validate_dimension_keys("reasons", grade.reasons, dimension_ids, errors)
    for dimension_id, reason in grade.reasons.items():
        if not reason.strip():
            errors.append(f"reasons.{dimension_id}: expected non-empty text")
        elif len(reason) > MAX_JUDGE_REASON_LENGTH:
            errors.append(f"reasons.{dimension_id}: expected concise text")


def _validate_evidence(
    grade: JudgeGrade,
    dimension_ids: set[str],
    errors: list[str],
) -> None:
    _validate_dimension_keys("evidence", grade.evidence, dimension_ids, errors)
    for dimension_id, evidence in grade.evidence.items():
        if not evidence.strip():
            errors.append(f"evidence.{dimension_id}: expected observable text")


def _validate_verdict(
    grade: JudgeGrade,
    errors: list[str],
) -> None:
    if grade.verdict not in {EvalVerdict.PASS, EvalVerdict.FAIL}:
        errors.append("verdict: expected pass or fail")
    if not 0.0 <= grade.confidence <= 1.0:
        errors.append("confidence: expected 0 through 1")


def _validate_dimension_keys(
    field_name: str,
    values: Mapping[str, object],
    dimension_ids: set[str],
    errors: list[str],
) -> None:
    actual_ids = set(values)
    for dimension_id in sorted(dimension_ids - actual_ids):
        errors.append(f"{field_name}.{dimension_id}: missing dimension")
    for dimension_id in sorted(actual_ids - dimension_ids):
        errors.append(f"{field_name}.{dimension_id}: unknown dimension")


def _with_retry_metadata(
    grade: JudgeGrade,
    validation_errors: tuple[str, ...],
) -> JudgeGrade:
    return JudgeGrade(
        verdict=grade.verdict,
        scores=grade.scores,
        reasons=grade.reasons,
        confidence=grade.confidence,
        ambiguous=grade.ambiguous,
        error_message=grade.error_message,
        rubric_id=grade.rubric_id,
        rubric_version=grade.rubric_version,
        case_id=grade.case_id,
        evidence=grade.evidence,
        judge_model=grade.judge_model,
        response_hash=grade.response_hash,
        response_preview=grade.response_preview,
        validation_errors=validation_errors,
        retry_count=1,
    )


def _exception_grade(error: Exception, judge_model: str) -> JudgeGrade:
    error_type, error_message = sanitize_error(error)
    return JudgeGrade(
        verdict=EvalVerdict.JUDGE_ERROR,
        scores={},
        reasons={},
        confidence=0.0,
        ambiguous=True,
        error_message=f"{error_type}: {error_message}",
        judge_model=judge_model,
    )
