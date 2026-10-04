"""Reconciliation never repeats uncertain work or abandons verification."""

import asyncio
from dataclasses import replace

import pytest

from agent_team.application.workflow.task_verification_service import (
    TaskVerificationService,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.runtime.agent_cleanup_timeout_error import (
    AgentCleanupTimeoutError,
)
from agent_team.domain.runtime.agent_provider_timeout_error import (
    AgentProviderTimeoutError,
)
from agent_team.domain.runtime.agent_segment_timeout_error import (
    AgentSegmentTimeoutError,
)
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)
from agent_team.domain.workflow.task_status import TaskStatus
from tests.integration.runtime.run_completion import (
    blocking_completion_verifier as blocking_verifier_module,
)
from tests.integration.runtime.run_completion.completion_segment import (
    CompletionSegment,
)
from tests.integration.runtime.run_completion.run_completion_scenario import (
    RunCompletionScenario,
)
from tests.unit.fakes.runtime.fake_monotonic_clock import FakeMonotonicClock
from tests.unit.fakes.runtime.fake_watchdog_runtime import FakeWatchdogRuntime


class TestRunCompletionWatchdogs:
    """Observe actual owned work completion before workspace teardown."""

    def test_reconciliation_uses_remaining_cleanup_budget(
        self,
        run_completion_scenario: RunCompletionScenario,
        blocking_verifier: blocking_verifier_module.BlockingCompletionVerifier,
        verification_clock: FakeMonotonicClock,
    ) -> None:
        """Runtime drain and safe verification share one cleanup deadline."""
        scenario = run_completion_scenario
        profile = scenario.harness.profile_catalog.get_profile(
            scenario.task.role
        )

        def prepare_submission(task: AgentTask, run: AgentRunRecord) -> None:
            for operation in ("activate", "patch", "check", "submit"):
                scenario.runtime.operations.perform(
                    operation, task, profile, run
                )
            verification_clock.advance(2)

        runtime = FakeWatchdogRuntime(
            before_wait=prepare_submission,
            on_cancel=lambda: verification_clock.advance(6),
        )
        blocking_verifier.before_wait = lambda: verification_clock.advance(5)
        blocking_verifier.release_on_cancel = True
        harness = replace(
            scenario.harness,
            runtime=runtime,
            context_provider=None,
            task_verification_service=TaskVerificationService(
                repository=scenario.repository,
                verifier=blocking_verifier,
                audit_reader=scenario.audit,
            ),
            cancel_verification=blocking_verifier.cancel_pending_operations,
            clock=verification_clock,
            poll_interval_seconds=0.001,
        )
        task = replace(
            scenario.task,
            watchdogs=AgentWatchdogSettings(segment_timeout_seconds=1),
        )

        async def execute() -> None:
            with pytest.raises(AgentSegmentTimeoutError):
                await asyncio.wait_for(harness.execute(task), timeout=0.3)
            assert await asyncio.to_thread(
                blocking_verifier.finished.wait, 0.1
            )

        asyncio.run(execute())
        assert runtime.execute_calls == 1
        assert blocking_verifier.cancel_requested.is_set()
        assert scenario.audit.list_runs(limit=1)[0].termination_reason == (
            "segment_timeout"
        )

    def test_no_advancement_deadline_applies_during_verification(
        self,
        run_completion_scenario: RunCompletionScenario,
        blocking_verifier: blocking_verifier_module.BlockingCompletionVerifier,
        verification_clock: FakeMonotonicClock,
    ) -> None:
        """A threaded verifier cannot hide the active convergence deadline."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            CompletionSegment(("activate", "patch", "check", "submit")),
        )
        harness = replace(
            scenario.harness,
            task_verification_service=TaskVerificationService(
                repository=scenario.repository,
                verifier=blocking_verifier,
                audit_reader=scenario.audit,
            ),
            cancel_verification=blocking_verifier.cancel_pending_operations,
            clock=verification_clock,
            poll_interval_seconds=0.001,
        )

        async def execute() -> None:
            execution = asyncio.create_task(harness.execute(scenario.task))
            assert await asyncio.to_thread(blocking_verifier.started.wait, 0.5)
            verification_clock.advance(1801)
            try:
                assert await asyncio.to_thread(
                    blocking_verifier.cancel_requested.wait, 0.1
                )
            finally:
                blocking_verifier.release.set()
                if not blocking_verifier.cancel_requested.is_set():
                    execution.cancel()
            with pytest.raises(AgentStalledError, match="deadline"):
                await execution
            assert blocking_verifier.finished.is_set()

        asyncio.run(execute())
        assert scenario.audit.list_runs(limit=1)[0].termination_reason == (
            "stalled"
        )

    @pytest.mark.parametrize(
        "failure",
        (
            (AgentProviderTimeoutError, "provider_timeout"),
            (AgentSegmentTimeoutError, "segment_timeout"),
            (AgentCleanupTimeoutError, "cleanup_timeout"),
        ),
    )
    @pytest.mark.parametrize(
        ("operations", "status", "revision"),
        (
            ((), TaskStatus.PENDING, 0),
            (("activate",), TaskStatus.IN_PROGRESS, 0),
            (("activate", "patch"), TaskStatus.IN_PROGRESS, 1),
            (("activate", "patch", "check"), TaskStatus.IN_PROGRESS, 1),
            (
                ("activate", "patch", "check", "submit"),
                TaskStatus.COMPLETED,
                1,
            ),
            (("activate", "block"), TaskStatus.BLOCKED, 0),
        ),
        ids=(
            "before-activation",
            "before-patch",
            "after-patch",
            "before-handoff",
            "after-handoff",
            "after-blocking",
        ),
    )
    def test_timeout_reconciles_persisted_state_without_replay(
        self,
        run_completion_scenario: RunCompletionScenario,
        failure: tuple[type[Exception], str],
        operations: tuple[str, ...],
        status: TaskStatus,
        revision: int,
    ) -> None:
        """Inspect durable effects on either side of mutation boundaries."""
        error_type, termination_reason = failure
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            CompletionSegment(
                operations, error=error_type("Uncertain runtime outcome.")
            ),
        )
        if status in {TaskStatus.COMPLETED, TaskStatus.BLOCKED}:
            result = asyncio.run(scenario.harness.execute(scenario.task))
            assert status.value in result.response
        else:
            with pytest.raises(error_type):
                asyncio.run(scenario.harness.execute(scenario.task))

        assert len(scenario.runtime.tasks) == 1
        assert scenario.runtime.operations.revision == revision
        assert scenario.task.task_id is not None
        current = scenario.repository.get_task(scenario.task.task_id)
        assert current is not None
        assert current.status is status
        assert len(scenario.verifier.submissions) == (
            1 if status is TaskStatus.COMPLETED else 0
        )
        run = scenario.audit.list_runs(limit=1)[0]
        invocations = scenario.audit.list_tool_invocations(run.id)
        assert len(invocations) == len(operations)
        assert run.feature_id == scenario.task.feature_id
        assert run.task_id == scenario.task.task_id
        assert run.role is scenario.task.role
        assert run.termination_reason == (
            f"task_{status.value}"
            if status in {TaskStatus.COMPLETED, TaskStatus.BLOCKED}
            else termination_reason
        )

    def test_failed_reconciliation_preserves_resumable_verification(
        self, run_completion_scenario: RunCompletionScenario
    ) -> None:
        """A failing safe verification never retries the timed-out segment."""
        scenario = run_completion_scenario
        scenario.verifier.required_revision = 2
        scenario.runtime.segments = (
            CompletionSegment(
                ("activate", "patch", "check", "submit"),
                error=AgentProviderTimeoutError("Provider timed out."),
            ),
        )

        with pytest.raises(AgentProviderTimeoutError):
            asyncio.run(scenario.harness.execute(scenario.task))

        assert len(scenario.runtime.tasks) == 1
        assert len(scenario.verifier.submissions) == 1
        assert scenario.task.task_id is not None
        task = scenario.repository.get_task(scenario.task.task_id)
        assert task is not None and task.status is TaskStatus.IN_PROGRESS
        evidence = scenario.repository.latest_task_verification(task.id)
        assert evidence is not None and evidence.outcome.value == "failed"
        assert scenario.audit.list_runs(limit=1)[0].termination_reason == (
            "provider_timeout"
        )

    def test_cancelled_verification_drains_actual_worker(
        self,
        run_completion_scenario: RunCompletionScenario,
        blocking_verifier: blocking_verifier_module.BlockingCompletionVerifier,
    ) -> None:
        """A cancelled asyncio wrapper is not a completed worker thread."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            CompletionSegment(("activate", "patch", "check", "submit")),
        )
        harness = replace(
            scenario.harness,
            task_verification_service=TaskVerificationService(
                repository=scenario.repository,
                verifier=blocking_verifier,
                audit_reader=scenario.audit,
            ),
            cancel_verification=blocking_verifier.cancel_pending_operations,
        )

        async def execute() -> None:
            execution = asyncio.create_task(harness.execute(scenario.task))
            assert await asyncio.to_thread(blocking_verifier.started.wait, 0.5)
            execution.cancel()
            assert await asyncio.to_thread(
                blocking_verifier.cancel_requested.wait, 0.5
            )
            try:
                done, _ = await asyncio.wait({execution}, timeout=0.01)
                assert not done
            finally:
                blocking_verifier.release.set()
            with pytest.raises(asyncio.CancelledError):
                await execution
            assert blocking_verifier.finished.is_set()

        asyncio.run(execute())
        assert scenario.task.task_id is not None
        task = scenario.repository.get_task(scenario.task.task_id)
        assert task is not None
        assert task.status is TaskStatus.VERIFICATION_PENDING
        assert scenario.repository.latest_task_verification(task.id) is None
        assert scenario.audit.list_runs(limit=1)[0].termination_reason == (
            "cancelled"
        )
