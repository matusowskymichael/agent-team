"""Tests for reviewed developer skill metadata and execution procedures."""

import pytest

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
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.infrastructure.skills.filesystem_agent_skill_catalog import (
    FilesystemAgentSkillCatalog,
)


class TestDeveloperSkills:
    """Keep reviewed developer knowledge aligned with verified task flow."""

    @pytest.mark.parametrize("area", ["backend", "frontend"])
    def test_developer_metadata_covers_available_submission(
        self,
        area: str,
    ) -> None:
        """Expose submission metadata within existing role capabilities."""
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole(f"{area}_developer"),
        )
        service = AgentSkillService(
            catalog=FilesystemAgentSkillCatalog(),
            authorizer=AgentSkillAuthorizer(),
        )

        metadata = service.list_available_metadata(profile)

        assert len(metadata) == 1
        assert metadata[0].name.value == f"implement-{area}-task"
        assert "submit_task_for_verification" in metadata[0].allowed_tools
        available_tools = {
            tool.value
            for tool in profile.allowed_tools | profile.allowed_workspace_tools
        }
        assert set(metadata[0].allowed_tools) <= available_tools

    @pytest.mark.parametrize("area", ["backend", "frontend"])
    def test_developer_procedure_orders_verified_completion(
        self,
        area: str,
    ) -> None:
        """Guard the observed activation, discovery, and handoff failures."""
        skill = FilesystemAgentSkillCatalog().load_skill(
            AgentSkillName(f"implement-{area}-task"),
        )

        steps = (
            "trusted assigned task",
            "bounded workspace discovery",
            "AuthService.logout",
            "`update_task_status`",
            "`apply_patch`",
            f'run_check(name="{area}")',
            "`submit_task_for_verification`",
            "final response after deterministic verification",
        )
        positions = [skill.body.index(step) for step in steps]
        assert positions == sorted(positions)
        assert "immediately call" in skill.body
        assert "known to be absent" in skill.body
        assert "repair the implementation" in skill.body
        assert (
            "Runtime supplies `changed_paths` from audited successful patches"
        ) in skill.body
        assert "`checks_attempted` from audited completed checks" in skill.body
        assert "either argument yourself" in skill.body
        assert "never grant additional tool access" in skill.body
