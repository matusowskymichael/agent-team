"""Tests for the application orchestrator."""

import asyncio

from agent_team.application.runtime.orchestrator import Orchestrator
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask
from tests.unit.fakes.runtime.fake_agent_executor import FakeAgentExecutor


class TestOrchestrator:
    """Orchestrator behavior tests."""

    def test_run_delegates_to_injected_agent_executor(self) -> None:
        """Use the injected executor to complete a task."""
        task = AgentTask(prompt="Explain dependency inversion.")
        expected_result = AgentResult(response="Depend on abstractions.")
        agent_executor = FakeAgentExecutor(result=expected_result)
        orchestrator = Orchestrator(agent_executor=agent_executor)

        result = asyncio.run(orchestrator.run(task))

        assert result == expected_result
        assert agent_executor.received_task == task
