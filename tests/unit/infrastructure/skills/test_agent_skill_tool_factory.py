"""Tests for Agents SDK Agent Skill tools."""

import asyncio
import json
from typing import Any, cast

import pytest
from agents import FunctionTool, Tool
from agents.tool_context import ToolContext

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.skills.agent_skill_authorizer import (
    AgentSkillAuthorizer,
)
from agent_team.application.skills.agent_skill_service import (
    AgentSkillService,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.skills.agent_skill import AgentSkill
from agent_team.domain.skills.agent_skill_catalog import AgentSkillCatalog
from agent_team.domain.skills.agent_skill_metadata import AgentSkillMetadata
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.infrastructure.skills.agent_skill_tool_factory import (
    SKILL_SERVER_NAME,
    AgentSkillToolFactory,
)
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)


class _Catalog:
    def __init__(self) -> None:
        self.load_calls = 0
        self.resource_calls = 0

    def list_metadata(self) -> tuple[AgentSkillMetadata, ...]:
        return (_metadata("write-requirements-artifact"),)

    def load_skill(self, name: AgentSkillName) -> AgentSkill:
        self.load_calls += 1
        return AgentSkill(
            metadata=_metadata(name.value),
            body="Full skill body with procedure.",
        )

    def read_resource(
        self,
        skill_name: AgentSkillName,
        relative_path: str,
    ) -> tuple[str, str]:
        self.resource_calls += 1
        return f"{skill_name.value}:{relative_path}:content", "resource-hash"


class TestAgentSkillToolFactory:
    """AgentSkillToolFactory behavior tests."""

    def test_creates_tools_only_for_profiles_with_skills(self) -> None:
        """Expose skill tools only when the profile has assigned skills."""
        audit_repository = FakeAgentAuditRepository()
        factory = _factory(_Catalog(), audit_repository)

        analyst_tools = factory.create_tools(
            _profile(DevelopmentRole.BUSINESS_ANALYST),
            audit_repository.open_run(role=DevelopmentRole.BUSINESS_ANALYST),
        )
        qa_tools = factory.create_tools(
            _profile(DevelopmentRole.QA_ENGINEER),
            audit_repository.open_run(role=DevelopmentRole.QA_ENGINEER),
        )

        assert _tool_names(analyst_tools) == {
            "load_skill",
            "read_skill_resource",
        }
        assert qa_tools == []

    def test_load_skill_audits_hash_without_body(self) -> None:
        """Load a skill while auditing metadata, not full content."""
        catalog = _Catalog()
        audit_repository = FakeAgentAuditRepository()
        tool = _load_tool(catalog, audit_repository)

        result = _invoke(
            tool,
            '{"name":"write-requirements-artifact"}',
        )

        assert result["instructions"] == "Full skill body with procedure."
        invocation = audit_repository.tool_invocations[1]
        assert invocation.server_name == SKILL_SERVER_NAME
        assert invocation.classification.value == "read_only"
        assert invocation.status.value == "completed"
        assert "Full skill body" not in (invocation.result_preview or "")
        assert "content_hash" in (invocation.result_preview or "")
        assert catalog.load_calls == 1

    def test_unassigned_skill_is_denied_without_catalog_load(self) -> None:
        """Deny unknown profile skills before reading."""
        catalog = _Catalog()
        audit_repository = FakeAgentAuditRepository()
        tool = _load_tool(catalog, audit_repository)

        result = _invoke(
            tool,
            '{"name":"unassigned-skill"}',
        )

        assert "SKILL_ACCESS_DENIED" in str(result["error"])
        assert catalog.load_calls == 0
        invocation = audit_repository.tool_invocations[1]
        assert invocation.status.value == "denied"

    def test_audit_start_failure_prevents_skill_load(self) -> None:
        """Do not read files when the allowed load cannot be audited."""
        catalog = _Catalog()
        audit_repository = FakeAgentAuditRepository(
            fail_start_tool_invocation=True,
        )
        tool = _load_tool(catalog, audit_repository)

        with pytest.raises(RuntimeError, match="tool audit start failed"):
            _invoke(tool, '{"name":"write-requirements-artifact"}')

        assert catalog.load_calls == 0

    def test_read_resource_audits_hash_without_content(self) -> None:
        """Read a skill resource without logging its full content."""
        catalog = _Catalog()
        audit_repository = FakeAgentAuditRepository()
        tool = _resource_tool(catalog, audit_repository)

        result = _invoke(
            tool,
            json.dumps(
                {
                    "skill_name": "write-requirements-artifact",
                    "relative_path": "notes.md",
                },
            ),
        )

        assert result["content"] == (
            "write-requirements-artifact:notes.md:content"
        )
        invocation = audit_repository.tool_invocations[1]
        assert "write-requirements-artifact:notes.md:content" not in (
            invocation.result_preview or ""
        )
        assert "resource-hash" in (invocation.result_preview or "")


def _factory(
    catalog: _Catalog,
    audit_repository: FakeAgentAuditRepository,
) -> AgentSkillToolFactory:
    return AgentSkillToolFactory(
        service=AgentSkillService(
            catalog=cast("AgentSkillCatalog", catalog),
            authorizer=AgentSkillAuthorizer(),
        ),
        audit_repository=audit_repository,
    )


def _load_tool(
    catalog: _Catalog,
    audit_repository: FakeAgentAuditRepository,
) -> FunctionTool:
    return _tool(catalog, audit_repository, "load_skill")


def _resource_tool(
    catalog: _Catalog,
    audit_repository: FakeAgentAuditRepository,
) -> FunctionTool:
    return _tool(catalog, audit_repository, "read_skill_resource")


def _tool(
    catalog: _Catalog,
    audit_repository: FakeAgentAuditRepository,
    name: str,
) -> FunctionTool:
    factory = _factory(catalog, audit_repository)
    run = audit_repository.open_run(role=DevelopmentRole.BUSINESS_ANALYST)
    tools = factory.create_tools(
        _profile(DevelopmentRole.BUSINESS_ANALYST),
        run,
    )
    return next(
        cast("FunctionTool", tool) for tool in tools if tool.name == name
    )


def _invoke(tool: FunctionTool, arguments_json: str) -> dict[str, object]:
    result = asyncio.run(
        tool.on_invoke_tool(
            cast("ToolContext[Any]", object()),
            arguments_json,
        ),
    )
    assert isinstance(result, dict)
    return cast("dict[str, object]", result)


def _profile(role: DevelopmentRole) -> AgentProfile:
    return AgentProfileCatalog().get_profile(role)


def _tool_names(tools: list[Tool]) -> set[str]:
    return {tool.name for tool in tools}


def _metadata(name: str) -> AgentSkillMetadata:
    return AgentSkillMetadata(
        name=AgentSkillName(name),
        description=f"{name} description",
        content_hash=f"{name}-hash",
        version="0.1.0",
    )
