"""Fake agent executor for unit tests."""

from dataclasses import dataclass

from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask


@dataclass(slots=True)
class FakeAgentExecutor:
    """Agent executor fake that records the received task."""

    result: AgentResult
    received_task: AgentTask | None = None

    async def execute(self, task: AgentTask) -> AgentResult:
        """Record and return the configured result."""
        self.received_task = task
        return self.result
