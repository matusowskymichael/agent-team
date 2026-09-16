"""Fixtures for trusted workflow MCP submissions."""

import pytest

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.mcp.client.authorized_mcp_server import (
    AuthorizedMCPServer,
)
from agent_team.infrastructure.mcp.client.development_workflow_mcp_server_config import (  # noqa: E501
    DevelopmentWorkflowMCPServerConfig,
)
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.mcp.fake_mcp_server import FakeMCPServer
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


@pytest.fixture
def authorized_submission_server() -> AuthorizedMCPServer:
    """Bind a developer submission to a trusted feature, task, and run."""
    repository = FakeWorkflowRepository()
    feature = repository.create_feature(
        title="Feature",
        description="Description",
        status=FeatureStatus.DRAFT,
    )
    role = DevelopmentRole.BACKEND_DEVELOPER
    task = repository.create_task(
        feature_id=feature.id,
        title="Task",
        description="Description",
        assigned_role=role,
        status=TaskStatus.IN_PROGRESS,
    )
    audit_repository = FakeAgentAuditRepository()
    run = audit_repository.open_run(
        role=role,
        feature_id=feature.id,
        workspace_identity_hash="workspace-hash",
    )
    return AuthorizedMCPServer(
        delegate=FakeMCPServer(
            tool_names=[tool.value for tool in WorkflowToolName],
        ),
        config=DevelopmentWorkflowMCPServerConfig(
            profile=AgentProfileCatalog().get_profile(role),
            authorizer=CapabilityAuthorizer(repository=repository),
            audit_repository=audit_repository,
            run=run,
            bound_task_id=task.id,
        ),
    )
