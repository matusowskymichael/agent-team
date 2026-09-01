"""Workflow MCP server availability error."""


class WorkflowMCPUnavailableError(RuntimeError):
    """Raised when the local workflow MCP server cannot start."""
