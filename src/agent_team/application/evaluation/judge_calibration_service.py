"""Application service for judge calibration metrics."""

from dataclasses import dataclass

from agent_team.domain.evaluation.calibration_result import CalibrationResult
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.human_label import HumanLabel


@dataclass(frozen=True, slots=True)
class JudgeCalibrationService:
    """Compare judge grades with human rubric labels."""

    def calibrate(
        self,
        eval_result: EvalRunResult,
        labels: tuple[HumanLabel, ...],
    ) -> CalibrationResult:
        """Calculate agreement metrics for available human labels."""
        judge_by_case = {
            result.case_id: result.judge_grade
            for result in eval_result.case_results
            if result.judge_grade is not None
        }
        compared_labels = [
            label for label in labels if label.case_id in judge_by_case
        ]
        if not compared_labels:
            return CalibrationResult(0.0, {}, {}, 0.0, ())

        verdict_matches = 0
        ambiguous_count = 0
        score_matches: dict[str, int] = {}
        score_totals: dict[str, int] = {}
        score_errors: dict[str, int] = {}
        disagreements: list[str] = []

        for label in compared_labels:
            judge_grade = judge_by_case[label.case_id]
            if judge_grade.verdict is label.verdict:
                verdict_matches += 1
            else:
                disagreements.append(label.case_id)
            if judge_grade.ambiguous:
                ambiguous_count += 1
            for dimension_id, human_score in label.scores.items():
                judge_score = judge_grade.scores.get(dimension_id)
                if judge_score is None:
                    continue
                score_totals[dimension_id] = (
                    score_totals.get(dimension_id, 0) + 1
                )
                if judge_score == human_score:
                    score_matches[dimension_id] = (
                        score_matches.get(dimension_id, 0) + 1
                    )
                score_errors[dimension_id] = score_errors.get(
                    dimension_id, 0
                ) + abs(judge_score - human_score)

        total = len(compared_labels)
        return CalibrationResult(
            verdict_agreement=verdict_matches / total,
            dimension_exact_agreement=_rates(score_matches, score_totals),
            dimension_mean_absolute_error=_rates(score_errors, score_totals),
            judge_ambiguity_rate=ambiguous_count / total,
            disagreements=tuple(disagreements),
        )


def _rates(
    numerators: dict[str, int],
    denominators: dict[str, int],
) -> dict[str, float]:
    return {
        dimension_id: numerators.get(dimension_id, 0) / denominator
        for dimension_id, denominator in denominators.items()
        if denominator > 0
    }
