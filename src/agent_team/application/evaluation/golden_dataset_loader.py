"""Application service for validating golden evaluation datasets."""

from dataclasses import dataclass

from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_intent import EvalCaseIntent
from agent_team.domain.evaluation.eval_context_policy import (
    EvalContextPolicy,
)
from agent_team.domain.evaluation.eval_feature_fixture import (
    EvalFeatureFixture,
)
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workspace.workspace_tool_name import WorkspaceToolName

READ_ONLY_WORKFLOW_TOOLS = frozenset(
    {
        WorkflowToolName.GET_FEATURE,
        WorkflowToolName.GET_FEATURE_OVERVIEW,
        WorkflowToolName.LIST_FEATURES,
        WorkflowToolName.LIST_ARTIFACTS,
        WorkflowToolName.LIST_TASKS,
    },
)
READ_ONLY_WORKSPACE_TOOLS = frozenset(
    {
        WorkspaceToolName.LIST_FILES,
        WorkspaceToolName.SEARCH_CODE,
        WorkspaceToolName.READ_FILE,
        WorkspaceToolName.RUN_CHECK,
    },
)


@dataclass(frozen=True, slots=True)
class GoldenDatasetLoader:
    """Validate loaded golden cases and build an evaluation suite."""

    def build_suite(
        self,
        suite_id: str,
        dataset_hash: str,
        cases: tuple[EvalCase, ...],
    ) -> EvalSuite:
        """Return a validated immutable evaluation suite."""
        _require_unique_case_ids(cases)
        for case in cases:
            _validate_case(case)
        return EvalSuite(
            id=suite_id,
            cases=cases,
            dataset_hash=dataset_hash,
        )


def _require_unique_case_ids(cases: tuple[EvalCase, ...]) -> None:
    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            raise ValueError(f"Duplicate eval case ID: {case.id}.")
        seen_ids.add(case.id)


def _validate_case(case: EvalCase) -> None:
    if not case.id.strip():
        raise ValueError("Eval case ID must not be blank.")
    if not case.user_input.strip():
        raise ValueError(f"Eval case {case.id} has blank user input.")
    _validate_tools(case)
    _validate_error(case)
    _validate_artifacts(case)
    _validate_max_output_tokens(case)
    _validate_context_tool_consistency(case)


def _validate_tools(case: EvalCase) -> None:
    for tool_call in case.expected_tool_calls:
        _validate_tool_name(tool_call.name)
    for tool_name in case.forbidden_tool_calls:
        _validate_tool_name(tool_name)
    for trajectory in case.acceptable_tool_trajectories:
        _validate_trajectory(case.id, trajectory)


def _validate_trajectory(
    case_id: str,
    trajectory: ExpectedToolTrajectory,
) -> None:
    if (
        not trajectory.required_tool_calls
        and trajectory.optional_read_only_tool_calls
    ):
        raise ValueError(
            f"Eval case {case_id} has only optional tool calls.",
        )
    for tool_call in trajectory.required_tool_calls:
        _validate_tool_name(tool_call.name)
    for tool_name in trajectory.forbidden_tool_calls:
        _validate_tool_name(tool_name)
    for tool_name in trajectory.optional_read_only_tool_calls:
        tool = _validate_tool_name(tool_name)
        if tool not in READ_ONLY_WORKFLOW_TOOLS | READ_ONLY_WORKSPACE_TOOLS:
            raise ValueError(
                "Optional trajectory calls must be read-only tools.",
            )


def _validate_error(case: EvalCase) -> None:
    if (
        case.expected_error is not None
        and not case.expected_error.error_type.strip()
    ):
        raise ValueError(f"Eval case {case.id} has a blank error type.")


def _validate_tool_name(
    tool_name: str,
) -> WorkflowToolName | WorkspaceToolName:
    try:
        return WorkflowToolName(tool_name)
    except ValueError:
        return WorkspaceToolName(tool_name)


def _validate_artifacts(case: EvalCase) -> None:
    for feature in case.feature_fixtures:
        for artifact in feature.artifacts:
            ArtifactKind(artifact.kind)


def _validate_max_output_tokens(case: EvalCase) -> None:
    if case.max_output_tokens is not None and case.max_output_tokens < 1:
        raise ValueError(
            f"Eval case {case.id} max_output_tokens must be positive.",
        )


def _validate_context_tool_consistency(case: EvalCase) -> None:
    if case.intent is EvalCaseIntent.UNSPECIFIED:
        return
    _validate_tool_dispatch(case)
    _validate_context_only(case)
    _validate_outcome_grounding(case)


def _validate_tool_dispatch(case: EvalCase) -> None:
    if case.intent is not EvalCaseIntent.TOOL_DISPATCH:
        return
    if not case.expected_tool_calls and not case.acceptable_tool_trajectories:
        raise ValueError(
            f"Tool-dispatch eval case {case.id} has no required tool call.",
        )
    if _has_context_only_trajectory(case):
        raise ValueError(
            f"Tool-dispatch eval case {case.id} accepts context-only.",
        )
    if (
        case.objective_response_facts
        and _facts_available_in_context(case)
        and not _requires_unavailable_collection(case)
    ):
        raise ValueError(
            f"Tool-dispatch eval case {case.id} preloads target answer data.",
        )


def _validate_context_only(case: EvalCase) -> None:
    if not _has_context_only_trajectory(case):
        return
    if case.context_policy is EvalContextPolicy.NO_FEATURE_PRELOAD:
        raise ValueError(
            f"Eval case {case.id} accepts context-only without preloaded "
            "feature data.",
        )
    if _requires_unavailable_collection(case):
        raise ValueError(
            f"Eval case {case.id} accepts context-only while required "
            "collections are unavailable.",
        )
    if case.objective_response_facts and not _facts_available_in_context(case):
        raise ValueError(
            f"Eval case {case.id} accepts context-only but objective facts "
            "are not preloaded.",
        )


def _validate_outcome_grounding(case: EvalCase) -> None:
    if case.intent is not EvalCaseIntent.OUTCOME_GROUNDING:
        return
    if (
        case.objective_response_facts
        and _facts_available_in_context(case)
        and not _requires_unavailable_collection(case)
        and not _has_context_only_trajectory(case)
    ):
        raise ValueError(
            f"Outcome-grounding eval case {case.id} omits context-only.",
        )


def _has_context_only_trajectory(case: EvalCase) -> bool:
    return any(
        not trajectory.required_tool_calls
        for trajectory in case.acceptable_tool_trajectories
    )


def _facts_available_in_context(case: EvalCase) -> bool:
    context_text = _available_context_text(case)
    if not context_text:
        return False
    return all(
        fact.lower() in context_text for fact in case.objective_response_facts
    )


def _available_context_text(case: EvalCase) -> str:
    if case.context_policy is EvalContextPolicy.NO_FEATURE_PRELOAD:
        return ""
    feature = _scoped_feature(case)
    if feature is None:
        return ""
    values = [
        feature.title,
        feature.description,
        feature.status.value,
    ]
    if case.context_policy is EvalContextPolicy.STANDARD_FEATURE_CONTEXT:
        values.extend(
            artifact.content
            for artifact in feature.artifacts
            if artifact.kind in _visible_artifact_kinds(case)
        )
        if _task_data_preloaded(case):
            values.extend(task.title for task in feature.tasks)
            values.extend(task.description for task in feature.tasks)
            values.extend(task.status.value for task in feature.tasks)
    return " ".join(values).lower()


def _requires_unavailable_collection(case: EvalCase) -> bool:
    text = " ".join(
        (
            case.user_input,
            *case.objective_response_facts,
        ),
    ).lower()
    if "task" in text and not _task_data_preloaded(case):
        return True
    return _artifact_lookup_requested(text) and not _artifact_data_preloaded(
        case,
    )


def _artifact_lookup_requested(text: str) -> bool:
    return any(
        term in text
        for term in (
            "artifact",
            "requirements",
            "acceptance criteria",
        )
    )


def _artifact_data_preloaded(case: EvalCase) -> bool:
    return (
        case.context_policy is EvalContextPolicy.STANDARD_FEATURE_CONTEXT
        and _scoped_feature(case) is not None
    )


def _task_data_preloaded(case: EvalCase) -> bool:
    if case.context_policy is not EvalContextPolicy.STANDARD_FEATURE_CONTEXT:
        return False
    return case.active_role in _ROLES_WITH_ALL_TASK_CONTEXT


def _visible_artifact_kinds(case: EvalCase) -> frozenset[ArtifactKind]:
    return _VISIBLE_ARTIFACT_KINDS_BY_ROLE.get(
        case.active_role,
        frozenset(ArtifactKind),
    )


def _scoped_feature(case: EvalCase) -> EvalFeatureFixture | None:
    feature_id = case.feature_scope_id
    if feature_id is None and case.feature_fixtures:
        feature_id = case.feature_fixtures[0].id
    if feature_id is None:
        return None
    return next(
        (
            feature
            for feature in case.feature_fixtures
            if feature.id == feature_id
        ),
        None,
    )


_ROLES_WITH_ALL_TASK_CONTEXT = frozenset(
    {
        DevelopmentRole.DELIVERY_MANAGER,
        DevelopmentRole.SOFTWARE_ARCHITECT,
        DevelopmentRole.QA_ENGINEER,
        DevelopmentRole.CODE_REVIEWER,
    },
)

_VISIBLE_ARTIFACT_KINDS_BY_ROLE = {
    DevelopmentRole.BUSINESS_ANALYST: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
        },
    ),
    DevelopmentRole.SOFTWARE_ARCHITECT: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        },
    ),
    DevelopmentRole.BACKEND_DEVELOPER: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        },
    ),
    DevelopmentRole.FRONTEND_DEVELOPER: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        },
    ),
    DevelopmentRole.QA_ENGINEER: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
            ArtifactKind.TEST_REPORT,
        },
    ),
    DevelopmentRole.CODE_REVIEWER: frozenset(ArtifactKind),
    DevelopmentRole.DELIVERY_MANAGER: frozenset(ArtifactKind),
}
