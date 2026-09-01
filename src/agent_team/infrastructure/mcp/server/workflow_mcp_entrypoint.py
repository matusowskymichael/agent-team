"""Entrypoint for the development workflow MCP server."""

import logging
import sys

import anyio

from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.infrastructure.configuration.workflow_database_path import (
    load_workflow_database_path,
)
from agent_team.infrastructure.mcp.server.stdio_transport import (
    run_mcp_server_stdio,
)
from agent_team.infrastructure.mcp.server.workflow_mcp_server import (
    create_workflow_mcp_server,
)

from ...persistence.sqlite.workflow.sqlite_workflow_repository import (
    SQLiteWorkflowRepository,
)


def main() -> None:
    """Start the workflow MCP server over stdio."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    repository = SQLiteWorkflowRepository(load_workflow_database_path())
    service = WorkflowService(repository=repository)
    server = create_workflow_mcp_server(service)
    anyio.run(run_mcp_server_stdio, server)


if __name__ == "__main__":
    main()
