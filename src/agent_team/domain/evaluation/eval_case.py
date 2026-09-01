"""Evaluation case domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.eval_case_intent import EvalCaseIntent
from agent_team.domain.evaluation.eval_context_policy import (
    EvalContextPolicy,
)
from agent_team.domain.evaluation.eval_feature_fixture import (
    EvalFeatureFixture,
)
from agent_team.domain.evaluation.eval_session_fixture import (
    EvalSessionFixture,
)
from agent_team.domain.evaluation.eval_workspace_file_fixture import (
    EvalWorkspaceFileFixture,
)
from agent_team.domain.evaluation.expected_database_effect import (
    ExpectedDatabaseEffect,
)
from agent_team.domain.evaluation.expected_error import ExpectedError
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)
from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One golden evaluation case."""

    id: str
    name: str
    category: str
    severity: str
    active_role: DevelopmentRole
    feature_fixtures: tuple[EvalFeatureFixture, ...]
    prior_session_turns: tuple[str, ...]
    user_input: str
    expected_tool_calls: tuple[ExpectedToolCall, ...]
    forbidden_tool_calls: tuple[str, ...]
    expected_database_effects: tuple[ExpectedDatabaseEffect, ...]
    forbidden_database_effects: tuple[ExpectedDatabaseEffect, ...]
    required_response_facts: tuple[str, ...]
    forbidden_response_claims: tuple[str, ...]
    rubric_id: str
    note: str
    objective_response_facts: tuple[str, ...] = ()
    semantic_response_requirements: tuple[str, ...] = ()
    prohibited_objective_claims: tuple[str, ...] = ()
    acceptable_tool_trajectories: tuple[ExpectedToolTrajectory, ...] = ()
    expected_error: ExpectedError | None = None
    semantic_judge_required: bool = True
    intent: EvalCaseIntent = EvalCaseIntent.UNSPECIFIED
    context_policy: EvalContextPolicy = (
        EvalContextPolicy.STANDARD_FEATURE_CONTEXT
    )
    session_fixtures: tuple[EvalSessionFixture, ...] = ()
    requested_session_id: str | None = None
    feature_scope_id: int | None = None
    task_scope_id: int | None = None
    workspace_files: tuple[EvalWorkspaceFileFixture, ...] = ()
    adjudication_note: str = ""
    max_output_tokens: int | None = None
