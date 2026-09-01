"""Application service for reading audit records."""

from dataclasses import dataclass

from agent_team.domain.audit.agent_audit_reader import AgentAuditReader
from agent_team.domain.audit.agent_run_details import AgentRunDetails
from agent_team.domain.audit.agent_run_not_found_error import (
    AgentRunNotFoundError,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord


@dataclass(frozen=True, slots=True)
class AuditQueryService:
    """Read-only use cases for human audit inspection."""

    reader: AgentAuditReader

    def list_runs(self, limit: int) -> list[AgentRunRecord]:
        """Return recent agent runs."""
        if limit < 1:
            raise ValueError("limit must be greater than zero.")
        return self.reader.list_runs(limit)

    def show_run(self, run_id: int) -> AgentRunDetails:
        """Return one run and its tool invocations."""
        run = self.reader.get_run(run_id)
        if run is None:
            raise AgentRunNotFoundError(f"Agent run {run_id} was not found.")
        invocations = self.reader.list_tool_invocations(run_id)
        return AgentRunDetails(
            run=run,
            tool_invocations=tuple(invocations),
        )
