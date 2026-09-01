"""Tests for the development workflow MCP server factory."""

from pathlib import Path

from agents.mcp import MCPServerStdio

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.configuration.workflow_database_path import (
    AGENT_TEAM_DB_PATH_ENV,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_process_options as mcp_options,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_config as workflow_mcp_config,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_factory as workflow_mcp_factory,
)
from agent_team.infrastructure.mcp.client.authorized_mcp_server import (
    AuthorizedMCPServer,
)
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestDevelopmentWorkflowMcpServerFactory:
    """Development workflow MCP server factory tests."""

    def test_creates_stdio_server_with_expected_parameters(self) -> None:
        """Use the current Python executable to start the MCP module."""
        database_path = "workflow.db"
        current_directory = Path("custom-workspace")
        environment = {
            AGENT_TEAM_DB_PATH_ENV: database_path,
            "EXISTING_ENV": "kept",
        }

        server = _create_server(
            environ=environment,
            python_executable="/venv/bin/python",
            cwd=current_directory,
        )
        delegate = _stdio_delegate(server)

        assert delegate.params.command == "/venv/bin/python"
        expected_args = ["-m", workflow_mcp_factory.WORKFLOW_MCP_MODULE]
        expected_name = workflow_mcp_factory.DEVELOPMENT_WORKFLOW_MCP_NAME
        assert delegate.params.args == expected_args
        assert delegate.params.cwd == current_directory
        assert server.name == expected_name
        assert delegate.cache_tools_list is True
        assert server.use_structured_content is False

    def test_passes_database_path_environment_to_subprocess(self) -> None:
        """Propagate AGENT_TEAM_DB_PATH to the MCP subprocess."""
        database_path = "agent-team-test.db"

        server = _create_server(
            environ={AGENT_TEAM_DB_PATH_ENV: database_path},
            python_executable="/venv/bin/python",
        )
        delegate = _stdio_delegate(server)

        assert delegate.params.env is not None
        assert delegate.params.env[AGENT_TEAM_DB_PATH_ENV] == database_path

    def test_configures_role_tool_filter(self) -> None:
        """Configure SDK discovery filtering from the active profile."""
        server = _create_server(role=DevelopmentRole.BUSINESS_ANALYST)
        delegate = _stdio_delegate(server)

        assert delegate.tool_filter == {
            "allowed_tool_names": [
                "add_artifact",
                "get_feature",
                "get_feature_overview",
                "list_artifacts",
                "list_features",
                "list_tasks",
            ],
        }

    def test_passes_bound_feature_to_authorized_wrapper(self) -> None:
        """Keep feature-scoped authorization in the wrapper."""
        server = _create_server(
            role=DevelopmentRole.SOFTWARE_ARCHITECT,
            bound_feature_id=7,
        )

        assert server.bound_feature_id == 7

    def test_passes_bound_task_to_authorized_wrapper(self) -> None:
        """Keep task-scoped authorization in the wrapper."""
        server = _create_server(
            role=DevelopmentRole.BACKEND_DEVELOPER,
            bound_feature_id=7,
            bound_task_id=11,
        )

        assert server.bound_task_id == 11


def _create_server(  # noqa: PLR0913, PLR0917
    role: DevelopmentRole = DevelopmentRole.DELIVERY_MANAGER,
    environ: dict[str, str] | None = None,
    python_executable: str = "/venv/bin/python",
    cwd: Path | None = None,
    bound_feature_id: int | None = None,
    bound_task_id: int | None = None,
) -> AuthorizedMCPServer:
    profile = AgentProfileCatalog().get_profile(role)
    authorizer = CapabilityAuthorizer(repository=FakeWorkflowRepository())
    audit_repository = FakeAgentAuditRepository()
    run = audit_repository.open_run(
        role=role,
        feature_id=bound_feature_id,
    )
    return workflow_mcp_factory.create_development_workflow_mcp_server(
        workflow_mcp_config.DevelopmentWorkflowMCPServerConfig(
            profile=profile,
            authorizer=authorizer,
            audit_repository=audit_repository,
            run=run,
            bound_task_id=bound_task_id,
        ),
        process_options=mcp_options.DevelopmentWorkflowMCPProcessOptions(
            environ=environ,
            python_executable=python_executable,
            cwd=cwd,
        ),
    )


def _stdio_delegate(server: AuthorizedMCPServer) -> MCPServerStdio:
    assert isinstance(server.delegate, MCPServerStdio)
    return server.delegate
