"""Factory for the local development workflow MCP server."""

import os
from pathlib import Path

from agents.mcp import MCPServerStdio, MCPServerStdioParams
from agents.mcp.util import create_static_tool_filter

from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_process_options as mcp_options,
)
from agent_team.infrastructure.mcp.client.authorized_mcp_server import (
    AuthorizedMCPServer,
)
from agent_team.infrastructure.mcp.client.development_workflow_mcp_server_config import (  # noqa: E501
    DevelopmentWorkflowMCPServerConfig,
)

DEVELOPMENT_WORKFLOW_MCP_NAME = "development_workflow"
WORKFLOW_MCP_MODULE = (
    "agent_team.infrastructure.mcp.server.workflow_mcp_entrypoint"
)
WORKFLOW_MCP_USE_STRUCTURED_CONTENT = False
WORKFLOW_MCP_TIMEOUT_SECONDS = 10.0


def create_development_workflow_mcp_server(
    config: DevelopmentWorkflowMCPServerConfig,
    process_options: mcp_options.DevelopmentWorkflowMCPProcessOptions
    | None = None,
) -> AuthorizedMCPServer:
    """Create the stdio MCP server for local workflow tools."""
    options = (
        process_options or mcp_options.DevelopmentWorkflowMCPProcessOptions()
    )
    environment = dict(
        os.environ if options.environ is None else options.environ,
    )
    parameters = MCPServerStdioParams(
        command=options.python_executable,
        args=[
            "-m",
            WORKFLOW_MCP_MODULE,
        ],
        env=environment,
        cwd=Path.cwd() if options.cwd is None else options.cwd,
    )
    allowed_tool_names = sorted(
        tool.value for tool in config.profile.allowed_tools
    )
    delegate = MCPServerStdio(
        params=parameters,
        cache_tools_list=True,
        name=DEVELOPMENT_WORKFLOW_MCP_NAME,
        client_session_timeout_seconds=WORKFLOW_MCP_TIMEOUT_SECONDS,
        tool_filter=create_static_tool_filter(
            allowed_tool_names=allowed_tool_names,
        ),
        require_approval="never",
        use_structured_content=WORKFLOW_MCP_USE_STRUCTURED_CONTENT,
    )
    return AuthorizedMCPServer(
        delegate=delegate,
        config=config,
    )
