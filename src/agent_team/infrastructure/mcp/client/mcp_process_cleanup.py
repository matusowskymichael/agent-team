"""Optional ownership boundary for forced local MCP process cleanup."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MCPProcessCleanup(Protocol):
    """Terminate only the subprocess group owned by this MCP transport."""

    def force_cleanup(self) -> None:
        """Request immediate owned-process termination without awaiting."""
        ...
