"""Evaluation judge port."""

from typing import Protocol

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.judge_correction_request import (
    JudgeCorrectionRequest,
)
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.rubric import Rubric


class EvalJudge(Protocol):
    """Port for local LLM rubric judging."""

    async def grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
    ) -> JudgeGrade:
        """Grade one candidate result using a local rubric judge."""
        ...

    async def correct_grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
        correction: JudgeCorrectionRequest,
    ) -> JudgeGrade:
        """Correct invalid judge output using a local rubric judge."""
        ...
