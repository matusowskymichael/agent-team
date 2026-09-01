"""Agent executor port."""

from typing import Protocol

from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask


class AgentExecutor(Protocol):
    """A port for executing agent tasks."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute an agent task and return the final result."""
        ...
