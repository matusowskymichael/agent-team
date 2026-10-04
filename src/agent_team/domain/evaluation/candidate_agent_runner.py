"""Candidate agent runner port."""

from typing import Protocol

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_candidate_execution_context import (
    EvalCandidateExecutionContext,
)
from agent_team.domain.evaluation.eval_case import EvalCase


class CandidateAgentRunner(Protocol):
    """Port for running a candidate agent against one eval case."""

    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
        *,
        context: EvalCandidateExecutionContext | None = None,
    ) -> CandidateRunResult:
        """Run one isolated candidate evaluation case."""
        ...
