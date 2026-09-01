"""Application service for comparing evaluation runs."""

from dataclasses import dataclass

from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_comparison_result import (
    EvalComparisonResult,
)
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_verdict import EvalVerdict


@dataclass(frozen=True, slots=True)
class EvalComparisonService:
    """Compare baseline and candidate evaluation runs."""

    def compare(
        self,
        baseline: EvalRunResult,
        candidate: EvalRunResult,
        allow_mismatched_inputs: bool = False,
    ) -> EvalComparisonResult:
        """Return improvements, regressions, and unchanged cases."""
        if not allow_mismatched_inputs:
            _require_equivalent_inputs(baseline, candidate)

        baseline_results = _case_results(baseline)
        candidate_results = _case_results(candidate)
        case_ids = sorted(set(baseline_results) | set(candidate_results))
        improved: list[str] = []
        regressed: list[str] = []
        unchanged: list[str] = []
        deterministic_improved: list[str] = []
        deterministic_regressed: list[str] = []
        deterministic_uncomparable: list[str] = []
        semantic_improved: list[str] = []
        semantic_regressed: list[str] = []
        semantic_uncomparable: list[str] = []
        for case_id in case_ids:
            before_result = baseline_results.get(case_id)
            after_result = candidate_results.get(case_id)
            before = None if before_result is None else before_result.verdict
            after = None if after_result is None else after_result.verdict
            infrastructure_blocked = _infrastructure_blocked(
                before_result,
                after_result,
            )
            if infrastructure_blocked:
                unchanged.append(case_id)
            elif _general_failed(before) and not _general_failed(after):
                improved.append(case_id)
            elif not _general_failed(before) and _general_failed(after):
                regressed.append(case_id)
            else:
                unchanged.append(case_id)

            if before_result is not None and after_result is not None:
                if infrastructure_blocked:
                    deterministic_uncomparable.append(case_id)
                    semantic_change = "uncomparable"
                else:
                    _compare_deterministic(
                        before_result.deterministic_grade.passed,
                        after_result.deterministic_grade.passed,
                        case_id,
                        deterministic_improved,
                        deterministic_regressed,
                    )
                    semantic_change = _semantic_change(
                        before_result,
                        after_result,
                    )
                _record_semantic_change(
                    semantic_change,
                    case_id,
                    semantic_improved,
                    semantic_regressed,
                    semantic_uncomparable,
                )
        return EvalComparisonResult(
            baseline_id=baseline.id,
            candidate_id=candidate.id,
            improved_cases=tuple(improved),
            regressed_cases=tuple(regressed),
            unchanged_cases=tuple(unchanged),
            warnings=(),
            deterministic_improved_cases=tuple(deterministic_improved),
            deterministic_regressed_cases=tuple(deterministic_regressed),
            deterministic_uncomparable_cases=tuple(
                deterministic_uncomparable,
            ),
            semantic_improved_cases=tuple(semantic_improved),
            semantic_regressed_cases=tuple(semantic_regressed),
            semantic_uncomparable_cases=tuple(semantic_uncomparable),
        )


def _require_equivalent_inputs(
    baseline: EvalRunResult,
    candidate: EvalRunResult,
) -> None:
    if baseline.dataset_hash != candidate.dataset_hash:
        raise ValueError(
            "Dataset hashes differ; comparison is not equivalent.",
        )
    if baseline.rubric_hash != candidate.rubric_hash:
        raise ValueError("Rubric hashes differ; comparison is not equivalent.")


def _case_results(result: EvalRunResult) -> dict[str, EvalCaseResult]:
    return {
        case_result.case_id: case_result for case_result in result.case_results
    }


def _compare_deterministic(
    before_passed: bool,
    after_passed: bool,
    case_id: str,
    improved: list[str],
    regressed: list[str],
) -> None:
    if not before_passed and after_passed:
        improved.append(case_id)
    elif before_passed and not after_passed:
        regressed.append(case_id)


def _semantic_change(
    before: EvalCaseResult,
    after: EvalCaseResult,
) -> str:
    if not _has_valid_judge_grade(before) or not _has_valid_judge_grade(after):
        return "uncomparable"
    before_grade = before.judge_grade
    after_grade = after.judge_grade
    if before_grade is None or after_grade is None:
        return "uncomparable"
    before_passed = _semantic_passed(before_grade.verdict)
    after_passed = _semantic_passed(after_grade.verdict)
    if before_passed and not after_passed:
        return "regressed"
    if not before_passed and after_passed:
        return "improved"
    return "unchanged"


def _record_semantic_change(
    change: str,
    case_id: str,
    improved: list[str],
    regressed: list[str],
    uncomparable: list[str],
) -> None:
    if change == "improved":
        improved.append(case_id)
    elif change == "regressed":
        regressed.append(case_id)
    elif change == "uncomparable":
        uncomparable.append(case_id)


def _semantic_passed(verdict: EvalVerdict) -> bool:
    return verdict in {EvalVerdict.PASSED, EvalVerdict.PASS}


def _infrastructure_blocked(
    before: EvalCaseResult | None,
    after: EvalCaseResult | None,
) -> bool:
    return (
        before is not None
        and before.verdict is EvalVerdict.INFRASTRUCTURE_ERROR
    ) or (
        after is not None and after.verdict is EvalVerdict.INFRASTRUCTURE_ERROR
    )


def _has_valid_judge_grade(result: EvalCaseResult) -> bool:
    grade = result.judge_grade
    if grade is None:
        return False
    return grade.verdict in {EvalVerdict.PASS, EvalVerdict.FAIL}


def _general_failed(verdict: EvalVerdict | None) -> bool:
    return verdict in {
        EvalVerdict.FAIL,
        EvalVerdict.DETERMINISTIC_FAILED,
        EvalVerdict.JUDGE_FAILED,
        EvalVerdict.JUDGE_ERROR,
        EvalVerdict.AMBIGUOUS,
    }
