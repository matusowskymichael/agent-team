"""Public capability boundary tests independent of discovery metadata."""

from dataclasses import replace

import pytest

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.domain.runtime.capability_denied_error import (
    CapabilityDeniedError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestCapabilityAuthorizerBoundaries:
    """Preserve trusted assignment and attribution across logical segments."""

    @pytest.mark.parametrize(
        ("role", "task_id", "feature_id", "binding", "allowed"),
        (
            (DevelopmentRole.BACKEND_DEVELOPER, 4, 2, 4, True),
            (DevelopmentRole.BACKEND_DEVELOPER, 4, 2, None, False),
            (DevelopmentRole.BACKEND_DEVELOPER, 4, 2, 9, False),
            (DevelopmentRole.BACKEND_DEVELOPER, 9, 2, 9, False),
            (DevelopmentRole.FRONTEND_DEVELOPER, 4, 2, 4, False),
            (DevelopmentRole.BACKEND_DEVELOPER, 4, 8, 4, False),
            (DevelopmentRole.SOFTWARE_ARCHITECT, 4, 2, 4, False),
        ),
    )
    def test_submission_requires_trusted_task_feature_and_role(  # noqa: PLR0913, PLR0917
        self,
        capability_repository: FakeWorkflowRepository,
        role: DevelopmentRole,
        task_id: int,
        feature_id: int,
        binding: int | None,
        allowed: bool,
    ) -> None:
        """Even widened discovery cannot grant cross-role task submission."""
        profile = AgentProfileCatalog().get_profile(role)
        profile = replace(
            profile,
            allowed_tools=frozenset(
                {WorkflowToolName.SUBMIT_TASK_FOR_VERIFICATION}
            ),
        )
        authorizer = CapabilityAuthorizer(capability_repository)
        if allowed:
            authorizer.authorize(
                profile,
                "submit_task_for_verification",
                {"task_id": task_id},
                feature_id,
                binding,
            )
        else:
            with pytest.raises(CapabilityDeniedError):
                authorizer.authorize(
                    profile,
                    "submit_task_for_verification",
                    {"task_id": task_id},
                    feature_id,
                    binding,
                )

    @pytest.mark.parametrize(
        ("task_id", "feature_id", "allowed"),
        ((4, 2, True), (9, 2, False), (9, None, False)),
    )
    def test_status_changes_require_an_existing_assigned_task(
        self,
        capability_repository: FakeWorkflowRepository,
        task_id: int,
        feature_id: int | None,
        allowed: bool,
    ) -> None:
        """Absent tasks cannot regain mutation authority after compaction."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER
        )
        authorizer = CapabilityAuthorizer(capability_repository)
        arguments: dict[str, object] = {
            "task_id": task_id,
            "status": "in_progress",
        }
        if allowed:
            authorizer.authorize(
                profile, "update_task_status", arguments, feature_id, task_id
            )
        else:
            with pytest.raises(CapabilityDeniedError):
                authorizer.authorize(
                    profile,
                    "update_task_status",
                    arguments,
                    feature_id,
                    task_id,
                )

    @pytest.mark.parametrize("status", (None, 1, "unknown"))
    def test_optional_feature_status_is_typed_and_enum_validated(
        self,
        capability_repository: FakeWorkflowRepository,
        status: object,
    ) -> None:
        """Accept an unset filter while rejecting malformed selection."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.DELIVERY_MANAGER
        )
        authorizer = CapabilityAuthorizer(capability_repository)
        if status is None:
            authorizer.authorize(profile, "list_features", {"status": status})
        else:
            with pytest.raises(CapabilityDeniedError):
                authorizer.authorize(
                    profile, "list_features", {"status": status}
                )

    @pytest.mark.parametrize(
        ("role", "status", "allowed"),
        (
            (DevelopmentRole.DELIVERY_MANAGER, "pending", True),
            (DevelopmentRole.BACKEND_DEVELOPER, "pending", False),
            (DevelopmentRole.SOFTWARE_ARCHITECT, None, False),
        ),
    )
    def test_task_creation_keeps_role_and_initial_status_policy(
        self,
        capability_repository: FakeWorkflowRepository,
        role: DevelopmentRole,
        status: object,
        allowed: bool,
    ) -> None:
        """Tool metadata cannot override role restrictions or status types."""
        profile = replace(
            AgentProfileCatalog().get_profile(role),
            allowed_tools=frozenset({WorkflowToolName.CREATE_TASK}),
        )
        arguments: dict[str, object] = {
            "feature_id": 2,
            "assigned_role": "backend_developer",
            "status": status,
        }
        authorizer = CapabilityAuthorizer(capability_repository)
        if allowed:
            authorizer.authorize(profile, "create_task", arguments, 2)
        else:
            with pytest.raises(CapabilityDeniedError):
                authorizer.authorize(profile, "create_task", arguments, 2)

    @pytest.mark.parametrize(
        "arguments",
        (
            None,
            {
                "feature_id": 2,
                "kind": "architecture",
                "content": "Plan.",
                "created_by": "agent:backend_developer",
            },
        ),
    )
    def test_artifact_mutation_rejects_missing_input_or_supplied_actor(
        self,
        capability_repository: FakeWorkflowRepository,
        arguments: dict[str, object] | None,
    ) -> None:
        """Keep artifact attribution under trusted runtime control."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT
        )
        with pytest.raises(CapabilityDeniedError):
            CapabilityAuthorizer(capability_repository).authorize(
                profile, "add_artifact", arguments, 2
            )
