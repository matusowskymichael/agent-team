"""Cancellable runtime fake for hanging execution regression tests."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask


@dataclass(slots=True)
class FakeWatchdogRuntime:
    """Wait forever until cancelled after an optional trusted side effect."""

    started: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_release: asyncio.Event = field(default_factory=asyncio.Event)
    slow_cleanup: bool = False
    before_wait: Callable[[AgentTask, AgentRunRecord], None] | None = None
    on_cancel: Callable[[], None] | None = None
    execute_calls: int = 0
    model_name: str = "fake-watchdog-model"

    async def execute(
        self,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
        context: AgentContextEnvelope | None = None,
        skill_context: str | None = None,
    ) -> AgentResult:
        """Record the trusted binding and expose deterministic cancellation."""
        assert profile.role is task.role
        assert context is None
        assert skill_context is None
        self.execute_calls += 1
        if self.before_wait is not None:
            self.before_wait(task, run)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            if self.on_cancel is not None:
                self.on_cancel()
            if self.slow_cleanup:
                await self.cleanup_release.wait()
            raise
        raise AssertionError("A hanging runtime must not finish normally.")
