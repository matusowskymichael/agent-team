"""Capability authorization for workflow tool calls."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.capability_denied_error import (
    CapabilityDeniedError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.workflow_repository import WorkflowRepository

DENIED_MESSAGE = "The requested workflow capability is not allowed."

ARTIFACT_KIND_ALLOWLIST = {
    DevelopmentRole.BUSINESS_ANALYST: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
        },
    ),
    DevelopmentRole.SOFTWARE_ARCHITECT: frozenset(
        {
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        },
    ),
    DevelopmentRole.QA_ENGINEER: frozenset({ArtifactKind.TEST_REPORT}),
    DevelopmentRole.CODE_REVIEWER: frozenset({ArtifactKind.CODE_REVIEW}),
}

TASK_ASSIGNMENT_RESTRICTED_ROLES = frozenset(
    {
        DevelopmentRole.BACKEND_DEVELOPER,
        DevelopmentRole.FRONTEND_DEVELOPER,
        DevelopmentRole.QA_ENGINEER,
        DevelopmentRole.CODE_REVIEWER,
    },
)

ARCHITECT_TASK_ASSIGNMENT_ROLES = frozenset(
    {
        DevelopmentRole.BACKEND_DEVELOPER,
        DevelopmentRole.FRONTEND_DEVELOPER,
        DevelopmentRole.QA_ENGINEER,
        DevelopmentRole.CODE_REVIEWER,
    },
)

FEATURE_ID_AUTHORIZATION_TOOLS = frozenset(
    {
        WorkflowToolName.GET_FEATURE,
        WorkflowToolName.GET_FEATURE_OVERVIEW,
        WorkflowToolName.LIST_ARTIFACTS,
        WorkflowToolName.LIST_TASKS,
    },
)


@dataclass(frozen=True, slots=True)
class CapabilityAuthorizer:
    """Authorize workflow tool calls for role profiles."""

    repository: WorkflowRepository

    def authorize(
        self,
        profile: AgentProfile,
        tool_name: str,
        arguments: Mapping[str, object] | None,
        bound_feature_id: int | None = None,
        bound_task_id: int | None = None,
    ) -> None:
        """Raise when a role is not authorized for a tool call."""
        tool = _parse_tool_name(tool_name)
        if tool not in profile.allowed_tools:
            _deny(f"The {profile.role.value} role cannot call {tool.value}.")

        if tool is WorkflowToolName.LIST_FEATURES:
            _validate_optional_feature_status(arguments)
        elif tool in FEATURE_ID_AUTHORIZATION_TOOLS:
            feature_id = _require_integer_argument(arguments, "feature_id")
            _validate_bound_feature_id(
                profile,
                feature_id,
                bound_feature_id,
            )
        elif tool is WorkflowToolName.CREATE_FEATURE:
            pass
        elif tool is WorkflowToolName.ADD_ARTIFACT:
            self._authorize_add_artifact(
                profile,
                arguments,
                bound_feature_id,
            )
        elif tool is WorkflowToolName.CREATE_TASK:
            self._authorize_create_task(profile, arguments, bound_feature_id)
        elif tool is WorkflowToolName.UPDATE_TASK_STATUS:
            self._authorize_update_task_status(
                profile,
                arguments,
                bound_feature_id,
                bound_task_id,
            )
        else:
            _deny()

    def _authorize_add_artifact(
        self,
        profile: AgentProfile,
        arguments: Mapping[str, object] | None,
        bound_feature_id: int | None,
    ) -> None:
        _deny_user_supplied_actor(profile, arguments)
        if profile.role is DevelopmentRole.SOFTWARE_ARCHITECT:
            _deny_unknown_arguments(
                arguments,
                {"feature_id", "kind", "content"},
            )
        feature_id = _require_integer_argument(arguments, "feature_id")
        _validate_bound_feature_id(profile, feature_id, bound_feature_id)
        artifact_kind = _require_artifact_kind(arguments)
        if profile.role is DevelopmentRole.DELIVERY_MANAGER:
            return
        allowed_kinds = ARTIFACT_KIND_ALLOWLIST.get(profile.role)
        if allowed_kinds is None or artifact_kind not in allowed_kinds:
            _deny(
                f"The {profile.role.value} role cannot create "
                f"{artifact_kind.value} artifacts.",
            )

    def _authorize_create_task(
        self,
        profile: AgentProfile,
        arguments: Mapping[str, object] | None,
        bound_feature_id: int | None,
    ) -> None:
        if profile.role is DevelopmentRole.SOFTWARE_ARCHITECT:
            _deny_unknown_arguments(
                arguments,
                {
                    "feature_id",
                    "title",
                    "description",
                    "assigned_role",
                    "status",
                },
            )
        feature_id = _require_integer_argument(arguments, "feature_id")
        _validate_bound_feature_id(profile, feature_id, bound_feature_id)
        assigned_role = _require_development_role(arguments, "assigned_role")
        task_status = _optional_task_status(arguments)
        if profile.role is DevelopmentRole.DELIVERY_MANAGER:
            return
        if profile.role is not DevelopmentRole.SOFTWARE_ARCHITECT:
            _deny(
                f"The {profile.role.value} role cannot create development "
                "tasks.",
            )
        if assigned_role not in ARCHITECT_TASK_ASSIGNMENT_ROLES:
            _deny(
                "The software_architect role cannot assign tasks to "
                f"{assigned_role.value}.",
            )
        if task_status is not None and task_status is not TaskStatus.PENDING:
            _deny(
                "The software_architect role must create tasks with the "
                "default pending status.",
            )

    def _authorize_update_task_status(
        self,
        profile: AgentProfile,
        arguments: Mapping[str, object] | None,
        bound_feature_id: int | None,
        bound_task_id: int | None,
    ) -> None:
        task_id = _require_integer_argument(arguments, "task_id")
        _require_task_status(arguments)
        if bound_task_id is not None and task_id != bound_task_id:
            _deny(
                f"The {profile.role.value} role cannot update task {task_id}.",
            )
        task = self.repository.get_task(task_id)
        if task is not None:
            _validate_bound_feature_id(
                profile,
                task.feature_id,
                bound_feature_id,
            )
        elif bound_feature_id is not None:
            _deny(
                f"The {profile.role.value} role cannot update task {task_id}.",
            )
        if profile.role is DevelopmentRole.DELIVERY_MANAGER:
            return
        if profile.role not in TASK_ASSIGNMENT_RESTRICTED_ROLES:
            _deny(
                f"The {profile.role.value} role cannot update task status.",
            )
        if task is None or task.assigned_role is not profile.role:
            _deny(
                f"The {profile.role.value} role cannot update task {task_id}.",
            )


def _parse_tool_name(tool_name: str) -> WorkflowToolName:
    try:
        return WorkflowToolName(tool_name)
    except ValueError as error:
        raise CapabilityDeniedError(
            "The requested tool is not a supported workflow capability.",
        ) from error


def _deny_user_supplied_actor(
    profile: AgentProfile,
    arguments: Mapping[str, object] | None,
) -> None:
    if arguments is not None and "created_by" in arguments:
        _deny(
            f"The {profile.role.value} role cannot provide created_by; "
            "actor identity is assigned by trusted runtime context.",
        )


def _validate_optional_feature_status(
    arguments: Mapping[str, object] | None,
) -> None:
    if arguments is None or "status" not in arguments:
        return
    value = arguments["status"]
    if value is None:
        return
    if not isinstance(value, str):
        _deny()
    _parse_enum(value, FeatureStatus)


def _require_artifact_kind(
    arguments: Mapping[str, object] | None,
) -> ArtifactKind:
    value = _require_string_argument(arguments, "kind")
    return _parse_enum(value, ArtifactKind)


def _require_task_status(arguments: Mapping[str, object] | None) -> TaskStatus:
    value = _require_string_argument(arguments, "status")
    return _parse_enum(value, TaskStatus)


def _optional_task_status(
    arguments: Mapping[str, object] | None,
) -> TaskStatus | None:
    if arguments is None or "status" not in arguments:
        return None
    value = arguments["status"]
    if not isinstance(value, str):
        _deny()
    return _parse_enum(value, TaskStatus)


def _require_development_role(
    arguments: Mapping[str, object] | None,
    name: str,
) -> DevelopmentRole:
    value = _require_string_argument(arguments, name)
    return _parse_enum(value, DevelopmentRole)


def _require_integer_argument(
    arguments: Mapping[str, object] | None,
    name: str,
) -> int:
    if arguments is None:
        _deny()
    value = arguments.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        _deny()
    return value


def _require_string_argument(
    arguments: Mapping[str, object] | None,
    name: str,
) -> str:
    if arguments is None:
        _deny()
    value = arguments.get(name)
    if not isinstance(value, str):
        _deny()
    return value


def _parse_enum[EnumValue: StrEnum](
    value: str,
    enum_type: type[EnumValue],
) -> EnumValue:
    try:
        return enum_type(value)
    except ValueError as error:
        raise CapabilityDeniedError(DENIED_MESSAGE) from error


def _deny(message: str = DENIED_MESSAGE) -> NoReturn:
    raise CapabilityDeniedError(message)


def _deny_unknown_arguments(
    arguments: Mapping[str, object] | None,
    allowed_names: set[str],
) -> None:
    if arguments is None:
        return
    unknown_names = sorted(set(arguments) - allowed_names)
    if unknown_names:
        _deny(
            "The requested workflow capability contains unsupported "
            f"arguments: {', '.join(unknown_names)}.",
        )


def _validate_bound_feature_id(
    profile: AgentProfile,
    feature_id: int,
    bound_feature_id: int | None,
) -> None:
    if bound_feature_id is None or feature_id == bound_feature_id:
        return
    _deny(
        f"The {profile.role.value} role cannot access feature {feature_id} "
        f"during a run bound to feature {bound_feature_id}.",
    )
