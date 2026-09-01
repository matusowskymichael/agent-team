"""Application orchestration for the agent team."""

from dataclasses import dataclass

from agent_team.domain.runtime.agent_executor import AgentExecutor
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask


@dataclass(frozen=True, slots=True)
class Orchestrator:
    """Application service for running agent tasks."""

    agent_executor: AgentExecutor

    async def run(self, task: AgentTask) -> AgentResult:
        """Run an agent task through the configured executor."""
        return await self.agent_executor.execute(task)
