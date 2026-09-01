"""Agent run details domain model."""

from dataclasses import dataclass

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord


@dataclass(frozen=True, slots=True)
class AgentRunDetails:
    """An agent run with its associated tool invocations."""

    run: AgentRunRecord
    tool_invocations: tuple[ToolInvocationRecord, ...]
