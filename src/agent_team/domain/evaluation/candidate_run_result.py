"""Candidate agent run result domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.database_effect import DatabaseEffect
from agent_team.domain.evaluation.eval_attempt_result import (
    EvalAttemptResult,
)
from agent_team.domain.evaluation.observed_skill_call import (
    ObservedSkillCall,
)
from agent_team.domain.evaluation.observed_tool_call import ObservedToolCall
from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    """Observable output from one candidate agent evaluation run."""

    role: DevelopmentRole
    model: str
    final_response: str
    tool_calls: tuple[ObservedToolCall, ...]
    database_effects: tuple[DatabaseEffect, ...]
    skill_calls: tuple[ObservedSkillCall, ...] = ()
    status: str = "completed"
    error_type: str | None = None
    error_message: str | None = None
    error_stage: str | None = None
    attempt_count: int = 1
    retry_count: int = 0
    attempts: tuple[EvalAttemptResult, ...] = ()
    max_output_tokens: int | None = None
