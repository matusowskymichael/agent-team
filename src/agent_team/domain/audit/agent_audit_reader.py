"""Agent audit reader port."""

from typing import Protocol

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord


class AgentAuditReader(Protocol):
    """Read-only port for local agent audit records."""

    def list_runs(self, limit: int) -> list[AgentRunRecord]:
        """Return recent agent runs up to the requested limit."""
        ...

    def get_run(self, run_id: int) -> AgentRunRecord | None:
        """Return one agent run by ID, if it exists."""
        ...

    def list_tool_invocations(
        self,
        run_id: int,
    ) -> list[ToolInvocationRecord]:
        """Return tool invocations for one agent run."""
        ...
