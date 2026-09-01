"""Tests for reviewed software architect Agent Skills."""

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.skills.agent_skill_authorizer import (
    AgentSkillAuthorizer,
)
from agent_team.application.skills.agent_skill_service import (
    AgentSkillService,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.infrastructure.skills.filesystem_agent_skill_catalog import (
    FilesystemAgentSkillCatalog,
)


class TestSoftwareArchitectSkills:
    """Software architect skill content and discovery tests."""

    def test_architect_skills_are_role_filtered(self) -> None:
        """Expose architect skills only through the architect profile."""
        service = AgentSkillService(
            catalog=FilesystemAgentSkillCatalog(),
            authorizer=AgentSkillAuthorizer(),
        )
        architect = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT,
        )
        analyst = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        architect_skills = service.list_available_metadata(architect)
        analyst_skills = service.list_available_metadata(analyst)

        assert {skill.name.value for skill in architect_skills} == {
            "review-architecture-readiness",
            "design-solution-architecture",
            "write-implementation-plan",
            "decompose-development-tasks",
        }
        assert "design-solution-architecture" not in {
            skill.name.value for skill in analyst_skills
        }

    def test_review_readiness_skill_is_read_only_and_grounded(self) -> None:
        """Require readiness reviews to inspect known inputs first."""
        skill = FilesystemAgentSkillCatalog().load_skill(
            AgentSkillName("review-architecture-readiness"),
        )

        assert "read-only workflow data" in skill.body
        assert "requirements and acceptance criteria" in skill.body
        assert "blockers" in skill.body
        assert "Ask targeted questions" in skill.body
        assert "never change role" in skill.body

    def test_design_architecture_skill_separates_save_from_preview(
        self,
    ) -> None:
        """Require explicit save intent for architecture artifacts."""
        skill = FilesystemAgentSkillCatalog().load_skill(
            AgentSkillName("design-solution-architecture"),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT,
        )

        assert "unsaved proposal" in skill.body
        assert "business facts" in skill.body
        assert "tradeoffs" in skill.body
        assert "Save only an `architecture` artifact" in skill.body
        assert "Never write application source code" in skill.body
        assert WorkflowToolName.ADD_ARTIFACT in profile.allowed_tools

    def test_implementation_plan_skill_requires_traceability(self) -> None:
        """Require implementation plans to stay grounded and code-free."""
        skill = FilesystemAgentSkillCatalog().load_skill(
            AgentSkillName("write-implementation-plan"),
        )

        assert "requirements, acceptance criteria" in skill.body
        assert "coherent phases" in skill.body
        assert "backend, frontend, QA" in skill.body
        assert "do not include source code" in skill.body
        assert "implementation_plan` artifact" in skill.body

    def test_task_decomposition_skill_limits_assignment_roles(self) -> None:
        """Require task creation only for permitted delivery roles."""
        skill = FilesystemAgentSkillCatalog().load_skill(
            AgentSkillName("decompose-development-tasks"),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT,
        )

        assert "Avoid duplicate tasks" in skill.body
        assert "`backend_developer`" in skill.body
        assert "`frontend_developer`" in skill.body
        assert "`qa_engineer`" in skill.body
        assert "`code_reviewer`" in skill.body
        assert "Never assign tasks to `business_analyst`" in skill.body
        assert WorkflowToolName.CREATE_TASK in profile.allowed_tools
