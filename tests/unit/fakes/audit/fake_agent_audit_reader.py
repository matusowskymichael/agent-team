"""Fake audit reader for unit tests."""

from dataclasses import dataclass, field

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord


def _empty_runs() -> list[AgentRunRecord]:
    return []


def _empty_tool_invocations() -> list[ToolInvocationRecord]:
    return []


def _empty_limits() -> list[int]:
    return []


@dataclass(slots=True)
class FakeAgentAuditReader:
    """In-memory fake for audit query tests."""

    runs: list[AgentRunRecord] = field(default_factory=_empty_runs)
    tool_invocations: list[ToolInvocationRecord] = field(
        default_factory=_empty_tool_invocations,
    )
    received_limits: list[int] = field(default_factory=_empty_limits)

    def list_runs(self, limit: int) -> list[AgentRunRecord]:
        """Return recent fake runs."""
        self.received_limits.append(limit)
        return self.runs[:limit]

    def get_run(self, run_id: int) -> AgentRunRecord | None:
        """Return one fake run by ID."""
        return next((run for run in self.runs if run.id == run_id), None)

    def list_tool_invocations(
        self,
        run_id: int,
    ) -> list[ToolInvocationRecord]:
        """Return fake tool invocations for one run."""
        return [
            invocation
            for invocation in self.tool_invocations
            if invocation.run_id == run_id
        ]
