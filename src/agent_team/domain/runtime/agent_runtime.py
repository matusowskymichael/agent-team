"""Agent runtime port."""

from typing import Protocol

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask


class AgentRuntime(Protocol):
    """Port for concrete model runtimes used by AgentHarness."""

    @property
    def model_name(self) -> str:
        """Return the configured runtime model name."""
        ...

    async def execute(
        self,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
        context: AgentContextEnvelope | None = None,
        skill_context: str | None = None,
    ) -> AgentResult:
        """Execute a task with a role-specific profile."""
        ...
