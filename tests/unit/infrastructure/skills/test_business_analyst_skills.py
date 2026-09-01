"""Tests for reviewed business analyst Agent Skills."""

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.infrastructure.skills.filesystem_agent_skill_catalog import (
    FilesystemAgentSkillCatalog,
)


class TestBusinessAnalystSkills:
    """Business analyst skill content tests."""

    def test_write_requirements_artifact_procedure_is_guarded(self) -> None:
        """Require content, trusted attribution, and allowed mutation path."""
        skill = FilesystemAgentSkillCatalog().load_skill(
            AgentSkillName("write-requirements-artifact"),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        assert "substantive requirements content" in skill.body
        assert "Ask for clarification" in skill.body
        assert "Do not invent placeholder content" in skill.body
        assert "created_by" in skill.body
        assert "trusted runtime context" in skill.body
        assert "requirements` only" in skill.body
        assert "Call `add_artifact` only after" in skill.body
        assert WorkflowToolName.ADD_ARTIFACT in profile.allowed_tools
        assert WorkflowToolName.CREATE_FEATURE not in profile.allowed_tools

    def test_write_acceptance_criteria_procedure_is_testable(self) -> None:
        """Require feature ID, testable wording, and no invented facts."""
        skill = FilesystemAgentSkillCatalog().load_skill(
            AgentSkillName("write-acceptance-criteria"),
        )

        assert "valid existing feature ID" in skill.body
        assert "observable, testable behavior" in skill.body
        assert "Ask for clarification" in skill.body
        assert "Do not invent business requirements" in skill.body
        assert "acceptance_criteria` only" in skill.body
        assert "explicitly asks" in skill.body

    def test_review_feature_readiness_is_read_only_and_grounded(self) -> None:
        """Require read-only grounded review behavior."""
        skill = FilesystemAgentSkillCatalog().load_skill(
            AgentSkillName("review-feature-readiness"),
        )

        assert "read-only workflow tools" in skill.body
        assert "Distinguish requirements" in skill.body
        assert "without inventing it" in skill.body
        assert "Remain read-only" in skill.body
        assert "Never infer artifact or task absence" in skill.body
        assert "Never access another feature" in skill.body
