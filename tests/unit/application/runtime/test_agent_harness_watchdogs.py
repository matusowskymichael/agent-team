"""Active runtime watchdogs and cancellation without mutation replay."""

import asyncio

import pytest

from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.agent_segment_timeout_error import (
    AgentSegmentTimeoutError,
)
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.runtime.fake_agent_runtime import FakeAgentRuntime
from tests.unit.fakes.runtime.fake_monotonic_clock import FakeMonotonicClock
from tests.unit.fakes.runtime.fake_watchdog_runtime import FakeWatchdogRuntime


class TestAgentHarnessWatchdogs:
    """A whole segment that never returns cannot defeat safety guards."""

    def test_explicit_watchdogs_survive_diagnostic_turn_override(self) -> None:
        """Changing the optional turn limit preserves custom safety limits."""
        audit = FakeAgentAuditRepository()
        runtime = FakeAgentRuntime(
            result=AgentResult("", segment_exhausted=True)
        )
        harness = AgentHarness(runtime=runtime, audit_repository=audit)
        task = AgentTask(
            prompt="Inspect the task.",
            run_limits=AgentRunLimits(max_turns=100),
            watchdogs=AgentWatchdogSettings(stagnant_segment_threshold=6),
        )

        with pytest.raises(AgentStalledError):
            asyncio.run(harness.execute(task))

        assert runtime.execute_calls == 6

    def test_cancellation_classification_survives_cleanup_deadline(
        self, watchdog_runtime: FakeWatchdogRuntime
    ) -> None:
        """Cleanup delay must not turn Ctrl+C into a segment timeout."""
        audit = FakeAgentAuditRepository()
        watchdog_runtime.slow_cleanup = True
        harness = AgentHarness(
            runtime=watchdog_runtime, audit_repository=audit
        )
        task = AgentTask(
            prompt="Inspect the task.",
            watchdogs=AgentWatchdogSettings(cleanup_grace_seconds=0.001),
        )

        async def execute() -> None:
            execution = asyncio.create_task(harness.execute(task))
            await watchdog_runtime.started.wait()
            execution.cancel()
            try:
                with pytest.raises(asyncio.CancelledError):
                    await execution
            finally:
                watchdog_runtime.cleanup_release.set()
            await asyncio.sleep(0)
            assert not [
                child
                for child in asyncio.all_tasks()
                if child is not asyncio.current_task() and not child.done()
            ]

        asyncio.run(execute())
        assert watchdog_runtime.execute_calls == 1
        assert audit.runs[1].termination_reason == "cancelled"
        assert "cleanup" in (audit.runs[1].error_message or "").lower()

    def test_runtime_segment_that_never_returns_is_cancelled(
        self, watchdog_runtime: FakeWatchdogRuntime
    ) -> None:
        """Cancel the owned runtime task and record a distinct timeout."""
        audit = FakeAgentAuditRepository()
        harness = AgentHarness(
            runtime=watchdog_runtime, audit_repository=audit
        )
        task = AgentTask(
            prompt="Inspect the task.",
            watchdogs=AgentWatchdogSettings(segment_timeout_seconds=0.001),
        )

        async def execute() -> None:
            with pytest.raises(AgentSegmentTimeoutError):
                await asyncio.wait_for(harness.execute(task), timeout=0.5)
            assert watchdog_runtime.cancelled.is_set()

        asyncio.run(execute())
        assert watchdog_runtime.execute_calls == 1
        assert audit.runs[1].status is AgentRunStatus.FAILED
        assert audit.runs[1].termination_reason == "segment_timeout"

    def test_active_runtime_obeys_no_advancement_clock(
        self,
        watchdog_runtime: FakeWatchdogRuntime,
        fake_monotonic_clock: FakeMonotonicClock,
    ) -> None:
        """The injected thirty-minute deadline applies inside a segment."""
        audit = FakeAgentAuditRepository()
        watchdog_runtime.before_wait = lambda _task, _run: (
            fake_monotonic_clock.advance(1801)
        )
        harness = AgentHarness(
            runtime=watchdog_runtime,
            audit_repository=audit,
            clock=fake_monotonic_clock,
            poll_interval_seconds=0.001,
        )

        async def execute() -> None:
            with pytest.raises(AgentStalledError, match="deadline"):
                await asyncio.wait_for(
                    harness.execute(AgentTask(prompt="Inspect the task.")),
                    timeout=0.5,
                )
            assert watchdog_runtime.cancelled.is_set()

        asyncio.run(execute())
        assert watchdog_runtime.execute_calls == 1
        assert audit.runs[1].termination_reason == "stalled"
