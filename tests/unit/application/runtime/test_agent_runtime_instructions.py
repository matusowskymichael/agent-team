"""Tests for runtime instruction generation."""

import pytest

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.agent_runtime_instructions import (
    build_runtime_instructions,
)
from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.capability_denied_error import (
    CapabilityDeniedError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestAgentRuntimeInstructions:
    """Runtime instruction generation behavior tests."""

    def test_business_analyst_instructions_are_capability_aware(
        self,
    ) -> None:
        """Describe available and prohibited analyst capabilities."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        instructions = build_runtime_instructions(profile)
        tools = _line_values(instructions, "Available tools:")
        artifact_kinds = _line_values(
            instructions,
            "Available artifact kinds:",
        )

        assert "Active role: business_analyst." in instructions
        assert "add_artifact" in tools
        assert "create_task" not in tools
        assert "requirements" in artifact_kinds
        assert "acceptance_criteria" in artifact_kinds
        assert "architecture" not in artifact_kinds
        assert "create architecture artifacts" in instructions
        assert "Never offer to perform an unavailable action." in instructions
        assert "Use get_feature_overview" in instructions
        assert "Capability denials are non-retryable" in instructions
        assert "Use an available skill" in instructions
        assert "Skills provide procedural knowledge only" in instructions
        assert "Available task assignment roles:" not in instructions

    @pytest.mark.parametrize("role", list(DevelopmentRole))
    def test_generated_instructions_list_only_profile_tools(
        self,
        role: DevelopmentRole,
    ) -> None:
        """List exactly the tools present in the immutable profile."""
        profile = AgentProfileCatalog().get_profile(role)

        instructions = build_runtime_instructions(profile)

        assert _line_values(instructions, "Available tools:") == {
            tool.value for tool in profile.allowed_tools
        }

    def test_business_analyst_profile_has_initial_skills(self) -> None:
        """Assign the first skills only to the business analyst role."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        assert {name.value for name in profile.allowed_skill_names} == {
            "write-requirements-artifact",
            "write-acceptance-criteria",
            "review-feature-readiness",
        }

    def test_software_architect_instructions_are_capability_aware(
        self,
    ) -> None:
        """Describe architect artifact and task assignment boundaries."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT,
        )

        instructions = build_runtime_instructions(profile)
        tools = _line_values(instructions, "Available tools:")
        artifact_kinds = _line_values(
            instructions,
            "Available artifact kinds:",
        )
        task_roles = _line_values(
            instructions,
            "Available task assignment roles:",
        )

        assert "You are the Software Architect specialist." in instructions
        assert "create_task" in tools
        assert "list_features" not in tools
        assert "update_task_status" not in tools
        assert artifact_kinds == {"architecture", "implementation_plan"}
        assert task_roles == {
            "backend_developer",
            "frontend_developer",
            "qa_engineer",
            "code_reviewer",
        }
        assert "create requirements artifacts" in instructions
        assert "assign tasks to business_analyst" in instructions
        assert "assign tasks to delivery_manager" in instructions
        assert "Do not write application source code" in instructions

    def test_bound_feature_context_instructions_are_immutable(
        self,
    ) -> None:
        """Tell feature-scoped agents to refuse cross-feature requests."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT,
        )
        context = AgentContextEnvelope(
            feature_id=1,
            session_id="architect-feature-1",
            authoritative_context="Feature 1 architecture context.",
            max_conversation_history_items=5,
        )

        instructions = build_runtime_instructions(profile, context=context)

        assert "Bound feature ID for this run: 1." in instructions
        assert "The bound feature ID is immutable." in instructions
        assert "refuse without calling any feature-scoped tool" in instructions
        assert "start a separate correctly bound session" in instructions

    def test_software_architect_profile_has_initial_skills(self) -> None:
        """Assign reviewed architect skills only to the architect role."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT,
        )

        assert {name.value for name in profile.allowed_skill_names} == {
            "review-architecture-readiness",
            "design-solution-architecture",
            "write-implementation-plan",
            "decompose-development-tasks",
        }

    def test_backend_developer_profile_has_workspace_tools(self) -> None:
        """Assign backend workspace tools, checks, prefixes, and skill."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        instructions = build_runtime_instructions(
            profile,
            task=AgentTask(
                prompt="Patch backend.",
                role=DevelopmentRole.BACKEND_DEVELOPER,
                feature_id=1,
                task_id=3,
            ),
        )

        assert {name.value for name in profile.allowed_skill_names} == {
            "implement-backend-task",
        }
        assert _line_values(instructions, "Available workspace tools:") == {
            "apply_patch",
            "list_files",
            "read_file",
            "run_check",
            "search_code",
        }
        assert "backend" in _line_values(
            instructions,
            "Allowed workspace path prefixes:",
        )
        assert "frontend" not in _line_values(
            instructions,
            "Allowed workspace path prefixes:",
        )
        assert "backend" in _line_values(
            instructions,
            "Available workspace checks:",
        )
        assert 'run_check(name="backend")' in instructions
        assert "Use individual ruff, pyright, or pytest checks only" in (
            instructions
        )
        assert "Trusted assigned task ID for this run: 3." in instructions
        assert "search for the proposed symbol name" in instructions

    def test_frontend_developer_profile_has_workspace_tools(self) -> None:
        """Assign frontend workspace tools, checks, prefixes, and skill."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.FRONTEND_DEVELOPER,
        )

        instructions = build_runtime_instructions(profile)

        assert {name.value for name in profile.allowed_skill_names} == {
            "implement-frontend-task",
        }
        assert "frontend" in _line_values(
            instructions,
            "Allowed workspace path prefixes:",
        )
        assert "backend" not in _line_values(
            instructions,
            "Allowed workspace path prefixes:",
        )
        assert _line_values(instructions, "Available workspace checks:") == {
            "frontend",
            "pytest",
            "ruff",
        }
        assert 'run_check(name="frontend")' in instructions
        assert "Use individual ruff or pytest checks only" in instructions

    @pytest.mark.parametrize(
        "role",
        [
            role
            for role in DevelopmentRole
            if role
            not in {
                DevelopmentRole.BUSINESS_ANALYST,
                DevelopmentRole.BACKEND_DEVELOPER,
                DevelopmentRole.FRONTEND_DEVELOPER,
                DevelopmentRole.SOFTWARE_ARCHITECT,
            }
        ],
    )
    def test_other_roles_have_no_skills(self, role: DevelopmentRole) -> None:
        """Avoid exposing unreviewed skills to other roles."""
        profile = AgentProfileCatalog().get_profile(role)

        assert profile.allowed_skill_names == frozenset()

    @pytest.mark.parametrize(
        "role",
        [
            DevelopmentRole.DELIVERY_MANAGER,
            DevelopmentRole.BUSINESS_ANALYST,
            DevelopmentRole.SOFTWARE_ARCHITECT,
            DevelopmentRole.QA_ENGINEER,
            DevelopmentRole.CODE_REVIEWER,
        ],
    )
    def test_non_developer_roles_have_no_workspace_tools(
        self,
        role: DevelopmentRole,
    ) -> None:
        """Expose workspace tools only to code-aware developers."""
        profile = AgentProfileCatalog().get_profile(role)

        assert profile.allowed_workspace_tools == frozenset()

    def test_runtime_instructions_include_skill_metadata_context(self) -> None:
        """Append only prebuilt skill metadata, not full skill bodies."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        instructions = build_runtime_instructions(
            profile,
            skill_context=(
                "Available skills:\n"
                "- write-requirements-artifact: Add requirements."
            ),
        )

        assert "write-requirements-artifact" in instructions
        assert "Add requirements." in instructions
        assert "Full skill body" not in instructions

    def test_authorization_remains_the_security_boundary(self) -> None:
        """Keep code authorization independent from instructions."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )
        authorizer = CapabilityAuthorizer(
            repository=FakeWorkflowRepository(),
        )

        with pytest.raises(CapabilityDeniedError):
            authorizer.authorize(
                profile=profile,
                tool_name="create_task",
                arguments={"feature_id": 1},
            )


def _line_values(instructions: str, prefix: str) -> set[str]:
    line = next(
        item for item in instructions.splitlines() if item.startswith(prefix)
    )
    values = line.removeprefix(prefix).strip().removesuffix(".")
    if values == "none":
        return set()
    return {value.strip() for value in values.split(",")}
