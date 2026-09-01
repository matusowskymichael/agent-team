"""Fake agent runtime for unit tests."""

from dataclasses import dataclass

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask


@dataclass(slots=True)
class FakeAgentRuntime:
    """Agent runtime fake that records the received task and profile."""

    result: AgentResult
    model_name_value: str = "fake-model"
    error: Exception | None = None
    received_task: AgentTask | None = None
    received_profile: AgentProfile | None = None
    received_run: AgentRunRecord | None = None
    received_context: AgentContextEnvelope | None = None
    received_skill_context: str | None = None
    execute_calls: int = 0

    @property
    def model_name(self) -> str:
        """Return the fake model name."""
        return self.model_name_value

    async def execute(
        self,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
        context: AgentContextEnvelope | None = None,
        skill_context: str | None = None,
    ) -> AgentResult:
        """Record and return the configured result."""
        self.execute_calls += 1
        self.received_task = task
        self.received_profile = profile
        self.received_run = run
        self.received_context = context
        self.received_skill_context = skill_context
        if self.error is not None:
            raise self.error
        return self.result
