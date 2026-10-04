"""Regressions for bounded candidate attempts and partial checkpoints."""

import asyncio
from collections.abc import Awaitable
from dataclasses import replace

import pytest

from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.application.evaluation.eval_candidate_state import (
    EvalCandidateState,
)
from agent_team.application.evaluation.eval_runner import EvalRunner
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.database_effect import DatabaseEffect
from agent_team.domain.evaluation.eval_candidate_execution_context import (
    EvalCandidateExecutionContext,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_timeout_error import (
    EvalCaseTimeoutError,
)
from agent_team.domain.evaluation.eval_progress_event import EvalProgressEvent
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)
from agent_team.domain.evaluation.eval_run_config import EvalRunConfig
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.runtime.agent_cleanup_timeout_error import (
    AgentCleanupTimeoutError,
)
from agent_team.domain.runtime.agent_lifecycle_phase import AgentLifecyclePhase
from agent_team.domain.runtime.agent_liveness_snapshot import (
    AgentLivenessSnapshot,
)
from agent_team.domain.runtime.agent_provider_timeout_error import (
    AgentProviderTimeoutError,
)
from agent_team.domain.runtime.agent_segment_timeout_error import (
    AgentSegmentTimeoutError,
)
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError


class MemoryCheckpointRepository:
    """Keep immutable checkpoints visible to public runner tests."""

    def __init__(self) -> None:
        """Start with an empty checkpoint history."""
        self.saved: list[EvalRunResult] = []

    def save(self, result: EvalRunResult) -> None:
        """Record a checkpoint without overwriting prior test evidence."""
        self.saved.append(result)

    def get(self, result_id: str) -> EvalRunResult | None:
        """Return the latest saved version of a run."""
        return next(
            (item for item in reversed(self.saved) if item.id == result_id),
            None,
        )

    def list_ids(self) -> list[str]:
        """Return the distinct identities recorded by the runner."""
        return sorted({item.id for item in self.saved})


class CancelOnCandidateRunner:
    """Cancel deterministically without any model or filesystem activity."""

    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
        *,
        context: object | None = None,
    ) -> CandidateRunResult:
        """Simulate explicit cancellation inside the active candidate."""
        _ = (case, candidate_model, repetition, context)
        raise asyncio.CancelledError


class WatchdogTestClock:
    """Advance safety deadlines deterministically without long sleeps."""

    def __init__(self) -> None:
        """Start at a zero monotonic timestamp."""
        self.current: float = 0.0

    def monotonic(self) -> float:
        """Return the injected monotonic timestamp."""
        return self.current


class ImmediateCaseDeadline:
    """Expire an active attempt by advancing a fake monotonic clock."""

    def __init__(self, clock: WatchdogTestClock) -> None:
        """Keep the injected clock and every independently applied deadline."""
        self.clock = clock
        self.timeouts: list[float] = []

    async def run[Result](
        self,
        operation: Awaitable[Result],
        timeout_seconds: float,
        cleanup_grace_seconds: float,
    ) -> Result:
        """Let partial effects finish, then request and drain cancellation."""
        assert 0 < cleanup_grace_seconds <= 10
        self.timeouts.append(timeout_seconds)
        task = asyncio.ensure_future(operation)
        await asyncio.sleep(0)
        self.clock.current += timeout_seconds
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise EvalCaseTimeoutError("Candidate-case safety deadline reached.")


class EvidenceUntilCancelled:
    """Capture safe partial evidence and wait for deadline cancellation."""

    def __init__(self) -> None:
        """Track attempts and whether each active child task was drained."""
        self.calls: int = 0
        self.finished: int = 0

    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
        *,
        context: EvalCandidateExecutionContext | None = None,
    ) -> CandidateRunResult:
        """Report durable effects without executing any real operation."""
        assert repetition > 0
        assert context is not None
        self.calls += 1
        snapshot = AgentLivenessSnapshot(
            segment_count=2,
            turns_used=12,
            task_status="in_progress",
            lifecycle_phase=AgentLifecyclePhase.IMPLEMENTED,
            last_tool_name="apply_patch",
            changed_paths=("backend/auth.py",),
            waiting_phase="model",
        )
        if context.liveness_observer is not None:
            context.liveness_observer.observe(snapshot)
        partial = CandidateRunResult(
            role=case.active_role,
            model=candidate_model,
            final_response="",
            tool_calls=(),
            database_effects=(
                DatabaseEffect(
                    "development_tasks",
                    "update",
                    {"id": 1, "status": "in_progress"},
                ),
            ),
            liveness_snapshot=snapshot,
            status="running",
        )
        if context.checkpoint is not None:
            context.checkpoint(partial)
        try:
            await asyncio.Event().wait()
        finally:
            self.finished += 1
        raise AssertionError("The test candidate must be cancelled.")


class CheckpointThenFailCandidate:
    """Record trusted partial effects before raising a runtime failure."""

    def __init__(self, error: Exception) -> None:
        """Retain the distinct failure classification for each attempt."""
        self.error = error
        self.calls = 0

    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
        *,
        context: EvalCandidateExecutionContext | None = None,
    ) -> CandidateRunResult:
        """Publish safely captured evidence before uncertain termination."""
        _ = repetition
        assert context is not None
        self.calls += 1
        partial = CandidateRunResult(
            role=case.active_role,
            model=candidate_model,
            final_response="",
            tool_calls=(),
            database_effects=(
                DatabaseEffect("development_tasks", "update", {"id": 1}),
            ),
        )
        if context.checkpoint is not None:
            context.checkpoint(partial)
        raise self.error


class HeartbeatSafetyReporter:
    """Release a waiting fake candidate once its live heartbeat arrives."""

    def __init__(self, released: asyncio.Event) -> None:
        """Record progress events and the candidate synchronization event."""
        self.released = released
        self.events: list[EvalProgressEvent] = []

    def report(self, event: EvalProgressEvent) -> None:
        """Observe progress through the existing reporter boundary."""
        self.events.append(event)
        if event.kind is EvalProgressEventKind.HEARTBEAT:
            self.released.set()


class HeartbeatSafetyCandidate:
    """Wait for one heartbeat while advancing only the injected clock."""

    def __init__(
        self, released: asyncio.Event, clock: WatchdogTestClock
    ) -> None:
        """Keep deterministic synchronization without a live model."""
        self.released = released
        self.clock = clock
        self.active_task_count = 0

    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
        *,
        context: EvalCandidateExecutionContext | None = None,
    ) -> CandidateRunResult:
        """Expose safe liveness while waiting for the progress observer."""
        _ = repetition
        assert context is not None
        assert context.liveness_observer is not None
        context.liveness_observer.observe(
            AgentLivenessSnapshot(segment_count=2, turns_used=12)
        )
        self.clock.current += 30
        self.active_task_count = len(asyncio.all_tasks())
        await self.released.wait()
        return CandidateRunResult(
            role=case.active_role,
            model=candidate_model,
            final_response="Advisory response.",
            tool_calls=(),
            database_effects=(),
        )


class TestEvalSafety:
    """Preserve interrupted runs and impose independent attempt deadlines."""

    def test_default_case_timeout_is_finite(self) -> None:
        """Every evaluation defaults to the explicit 45-minute fail-safe."""
        config = EvalRunConfig("local", "instructions")

        assert config.case_timeout_seconds == 2700

    def test_checkpoint_preserves_liveness_timing(
        self,
        golden_validation_case: EvalCase,
    ) -> None:
        """Merge final durable effects without discarding observed run ages."""
        clock = WatchdogTestClock()
        saved: list[CandidateRunResult] = []
        state = EvalCandidateState(
            checkpoint_callback=saved.append, clock=clock.monotonic
        )
        state.observe(
            AgentLivenessSnapshot(
                segment_count=2,
                turns_used=17,
                elapsed_seconds=700,
                time_since_advancement_seconds=300,
                task_status="in_progress",
                lifecycle_phase=AgentLifecyclePhase.CHECKED,
            )
        )
        clock.current += 25
        state.checkpoint(
            CandidateRunResult(
                role=golden_validation_case.active_role,
                model="local",
                final_response="",
                tool_calls=(),
                database_effects=(),
                liveness_snapshot=AgentLivenessSnapshot(
                    segment_count=2,
                    task_status="verification_pending",
                    lifecycle_phase=AgentLifecyclePhase.SUBMITTED,
                    last_tool_name="submit_task_for_verification",
                    changed_paths=("backend/auth.py",),
                    waiting_phase="cleanup",
                ),
            )
        )

        assert len(saved) == 1
        observed = saved[0].liveness_snapshot
        assert observed is not None
        assert observed.turns_used == 17
        assert observed.elapsed_seconds == 725
        assert observed.time_since_advancement_seconds == 325
        assert observed.task_status == "verification_pending"
        assert observed.lifecycle_phase is AgentLifecyclePhase.SUBMITTED

    def test_cancellation_checkpoints_before_propagation(
        self,
        eval_safety_suite: EvalSuite,
        eval_safety_rubric: Rubric,
    ) -> None:
        """Allocate an identity and persist an interrupted active case."""
        repository = MemoryCheckpointRepository()
        runner = EvalRunner(
            candidate_runner=CancelOnCandidateRunner(),
            grader=DeterministicEvalGrader(),
            result_repository=repository,
        )

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                runner.run_suite(
                    eval_safety_suite,
                    eval_safety_rubric,
                    EvalRunConfig("local", "instructions"),
                )
            )

        assert repository.saved[0].status == "running"
        assert repository.saved[-1].status == "interrupted"
        assert len(repository.list_ids()) == 1
        assert repository.saved[-1].active_case_id == "boundary-case"

    @pytest.mark.parametrize("limit", [2700.0, 5400.0])
    def test_timeout_preserves_attempt_without_infrastructure_retry(
        self,
        eval_safety_suite: EvalSuite,
        eval_safety_rubric: Rubric,
        limit: float,
    ) -> None:
        """Retain partial effects and move to the next case after timeout."""
        repository = MemoryCheckpointRepository()
        clock = WatchdogTestClock()
        candidate = EvidenceUntilCancelled()
        deadline = ImmediateCaseDeadline(clock)
        runner = EvalRunner(
            candidate_runner=candidate,
            grader=DeterministicEvalGrader(),
            result_repository=repository,
            clock=clock,
            deadline=deadline,
        )
        suite = replace(
            eval_safety_suite,
            cases=(
                *eval_safety_suite.cases,
                replace(eval_safety_suite.cases[0], id="second-case"),
            ),
        )

        result = asyncio.run(
            runner.run_suite(
                suite,
                eval_safety_rubric,
                EvalRunConfig(
                    "local",
                    "instructions",
                    case_timeout_seconds=limit,
                    infrastructure_retries=3,
                ),
            )
        )

        assert result.status == "timed_out"
        assert len(result.case_results) == 2
        assert candidate.calls == candidate.finished == 2
        assert deadline.timeouts == [limit, limit]
        assert result.duration_seconds == limit * 2
        assert len(repository.list_ids()) == 1
        for case in result.case_results:
            assert case.verdict is EvalVerdict.TIMED_OUT
            assert not case.deterministic_grade.passed
            observed = case.candidate_result
            assert observed.attempt_count == 1
            assert observed.retry_count == 0
            assert observed.status == "timed_out"
            assert observed.error_type == "EvalCaseTimeoutError"
            assert observed.termination_reason == "case_timeout"
            assert observed.database_effects
            assert observed.liveness_snapshot is not None
            assert observed.liveness_snapshot.segment_count == 2
            assert observed.liveness_snapshot.turns_used == 12
            assert observed.liveness_snapshot.elapsed_seconds == limit
        assert any(
            item.active_candidate is not None
            and item.active_candidate.status == "timed_out"
            and not item.case_results
            for item in repository.saved
        )

    @pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_candidate_deadline_is_rejected(
        self, limit: float
    ) -> None:
        """Require an enabled finite fail-safe for every candidate attempt."""
        with pytest.raises(ValueError, match="positive and finite"):
            EvalRunConfig("local", "instructions", case_timeout_seconds=limit)

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("Failed at /private/workspace api_key=private-key"),
            AgentProviderTimeoutError("Provider response timeout."),
            AgentSegmentTimeoutError("Runtime segment timeout."),
            AgentCleanupTimeoutError("Owned cleanup timeout."),
            AgentStalledError("No durable advancement."),
        ],
    )
    def test_runtime_failure_retains_partial_checkpoint_and_no_retry(
        self,
        eval_safety_suite: EvalSuite,
        eval_safety_rubric: Rubric,
        error: Exception,
    ) -> None:
        """Preserve effects when a boundary fails after checkpointing."""
        candidate = CheckpointThenFailCandidate(error)
        repository = MemoryCheckpointRepository()
        result = asyncio.run(
            EvalRunner(
                candidate,
                DeterministicEvalGrader(),
                result_repository=repository,
            ).run_suite(
                eval_safety_suite,
                eval_safety_rubric,
                EvalRunConfig(
                    "local", "instructions", infrastructure_retries=3
                ),
            )
        )

        assert candidate.calls == 1
        observed = result.case_results[0].candidate_result
        assert observed.database_effects
        assert observed.error_type == type(error).__name__
        assert observed.attempt_count == 1
        assert result.termination_type == type(error).__name__
        assert "private-key" not in (observed.error_message or "")
        assert "/private/workspace" not in (observed.error_message or "")
        if isinstance(error, AgentCleanupTimeoutError):
            assert result.status == "timed_out"
            assert observed.termination_reason == "cleanup_timeout"
            assert result.case_results[0].verdict is EvalVerdict.TIMED_OUT
        if isinstance(error, AgentStalledError):
            assert observed.status == "failed"
            assert observed.termination_reason == "stalled"

    def test_heartbeat_reports_live_clock_and_stops_after_candidate(
        self,
        eval_safety_suite: EvalSuite,
        eval_safety_rubric: Rubric,
    ) -> None:
        """Emit live diagnostics and drain heartbeat ownership at the end."""

        async def scenario() -> None:
            released = asyncio.Event()
            clock = WatchdogTestClock()
            reporter = HeartbeatSafetyReporter(released)
            runner = EvalRunner(
                HeartbeatSafetyCandidate(released, clock),
                DeterministicEvalGrader(),
                progress_reporter=reporter,
                clock=clock,
                heartbeat_interval_seconds=0.001,
            )
            await runner.run_suite(
                eval_safety_suite,
                eval_safety_rubric,
                EvalRunConfig("local", "instructions"),
            )
            heartbeat = next(
                event
                for event in reporter.events
                if event.kind is EvalProgressEventKind.HEARTBEAT
            )
            assert heartbeat.case_timeout_seconds == 2700
            assert heartbeat.case_remaining_seconds == 2670
            assert heartbeat.liveness_snapshot is not None
            assert heartbeat.liveness_snapshot.elapsed_seconds == 30
            assert heartbeat.liveness_snapshot.turns_used == 12
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(scenario())

    def test_no_progress_creates_no_heartbeat_task(
        self,
        eval_safety_suite: EvalSuite,
        eval_safety_rubric: Rubric,
    ) -> None:
        """Leave progress disabled through the reporter boundary."""

        async def scenario() -> None:
            released = asyncio.Event()
            released.set()
            clock = WatchdogTestClock()
            candidate = HeartbeatSafetyCandidate(released, clock)
            await EvalRunner(
                candidate, DeterministicEvalGrader(), clock=clock
            ).run_suite(
                eval_safety_suite,
                eval_safety_rubric,
                EvalRunConfig("local", "instructions"),
            )
            assert candidate.active_task_count == 2
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(scenario())
