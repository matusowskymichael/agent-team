"""Tests for Agent Skill application services."""

import pytest

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.skills.agent_skill_authorizer import (
    AgentSkillAuthorizer,
)
from agent_team.application.skills.agent_skill_context_builder import (
    AgentSkillContextBuilder,
)
from agent_team.application.skills.agent_skill_service import (
    AgentSkillService,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.skills.agent_skill import AgentSkill
from agent_team.domain.skills.agent_skill_access_denied_error import (
    AgentSkillAccessDeniedError,
)
from agent_team.domain.skills.agent_skill_metadata import AgentSkillMetadata
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)


class _Catalog:
    def __init__(self) -> None:
        self.load_calls = 0
        self.metadata = (
            _metadata("write-requirements-artifact"),
            _metadata("write-acceptance-criteria"),
            _metadata("review-feature-readiness"),
            _metadata("unassigned-skill"),
        )

    def list_metadata(self) -> tuple[AgentSkillMetadata, ...]:
        return self.metadata

    def load_skill(self, name: AgentSkillName) -> AgentSkill:
        self.load_calls += 1
        return AgentSkill(
            metadata=_metadata(name.value),
            body="Loaded body.",
        )

    def read_resource(
        self,
        skill_name: AgentSkillName,
        relative_path: str,
    ) -> tuple[str, str]:
        self.load_calls += 1
        return f"{skill_name.value}:{relative_path}", "resource-hash"


class TestAgentSkillService:
    """AgentSkillService behavior tests."""

    def test_business_analyst_metadata_is_role_filtered(self) -> None:
        """Return only skills assigned to the business analyst profile."""
        service = _service(_Catalog())
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        metadata = service.list_available_metadata(profile)

        assert {item.name.value for item in metadata} == {
            "write-requirements-artifact",
            "write-acceptance-criteria",
            "review-feature-readiness",
        }

    def test_other_roles_receive_no_skills(self) -> None:
        """Do not expose business analyst skills to other roles."""
        service = _service(_Catalog())
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT,
        )

        assert service.list_available_metadata(profile) == ()

    def test_unknown_skill_denial_does_not_touch_catalog(self) -> None:
        """Deny unassigned skills before loading files."""
        catalog = _Catalog()
        service = _service(catalog)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        with pytest.raises(AgentSkillAccessDeniedError):
            service.load_skill(profile, "unassigned-skill")

        assert catalog.load_calls == 0

    def test_malformed_skill_name_does_not_touch_catalog(self) -> None:
        """Reject malformed skill names before loading files."""
        catalog = _Catalog()
        service = _service(catalog)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        with pytest.raises(InvalidAgentSkillError):
            service.load_skill(profile, "../secret")

        assert catalog.load_calls == 0

    def test_allowed_tools_metadata_cannot_grant_permissions(self) -> None:
        """Ignore optional skill allowed-tools for workflow authorization."""
        service = _service(_Catalog())
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        metadata = service.list_available_metadata(profile)

        assert metadata[0].allowed_tools == ("create_feature",)
        assert WorkflowToolName.CREATE_FEATURE not in profile.allowed_tools

    def test_context_builder_lists_metadata_without_bodies(self) -> None:
        """Advertise skill names and descriptions only."""
        service = _service(_Catalog())
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        context = AgentSkillContextBuilder().build_context(
            service.list_available_metadata(profile),
        )

        assert "write-requirements-artifact" in context
        assert "Loaded body." not in context
        assert "hash" in context

    def test_context_builder_reports_no_skills(self) -> None:
        """Render an explicit no-skills context."""
        context = AgentSkillContextBuilder().build_context(())

        assert context == "Available skills: none."


def _service(catalog: _Catalog) -> AgentSkillService:
    return AgentSkillService(
        catalog=catalog,
        authorizer=AgentSkillAuthorizer(),
    )


def _metadata(name: str) -> AgentSkillMetadata:
    return AgentSkillMetadata(
        name=AgentSkillName(name),
        description=f"{name} description",
        content_hash=f"{name}-hash",
        version="0.1.0",
        allowed_tools=("create_feature",),
    )
