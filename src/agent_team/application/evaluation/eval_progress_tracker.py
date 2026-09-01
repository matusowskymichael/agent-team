"""Evaluation progress state tracker."""

import time
from dataclasses import dataclass, field, replace

from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_clock import EvalClock
from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_event import (
    EvalProgressEvent,
)
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)
from agent_team.domain.evaluation.eval_progress_reporter import (
    EvalProgressReporter,
)
from agent_team.domain.evaluation.eval_run_config import EvalRunConfig
from agent_team.domain.evaluation.eval_suite import EvalSuite


@dataclass(slots=True)
class EvalProgressTracker:
    """Track monotonic evaluation timing and emit progress events."""

    suite: EvalSuite
    config: EvalRunConfig
    reporter: EvalProgressReporter | None = None
    clock: EvalClock | None = None
    completed_cases: int = 0
    run_started_at: float = field(init=False)
    phase_durations: dict[EvalPhase, list[float]] = field(
        default_factory=dict[EvalPhase, list[float]],
    )

    def __post_init__(self) -> None:
        """Capture the monotonic start time."""
        self.run_started_at = self.monotonic()

    @property
    def total_cases(self) -> int:
        """Return the total selected case repetitions."""
        return len(self.suite.cases) * self.config.repetitions

    def monotonic(self) -> float:
        """Return the current monotonic timestamp."""
        if self.clock is None:
            return time.monotonic()
        return self.clock.monotonic()

    def elapsed_since(self, started_at: float) -> float:
        """Return a non-negative duration since a monotonic timestamp."""
        return max(0.0, self.monotonic() - started_at)

    def run_started(self) -> None:
        """Report that evaluation execution has started."""
        self._report(EvalProgressEventKind.RUN_STARTED)

    def phase_started(
        self,
        case: EvalCase,
        repetition: int,
        phase: EvalPhase,
    ) -> None:
        """Report that a case phase has started."""
        self._report(
            EvalProgressEventKind.PHASE_STARTED,
            case=case,
            repetition=repetition,
            phase=phase,
        )

    def phase_completed(
        self,
        case: EvalCase,
        repetition: int,
        phase: EvalPhase,
        duration_seconds: float,
        include_in_estimate: bool = True,
    ) -> None:
        """Record and report a completed case phase."""
        if include_in_estimate:
            self.phase_durations.setdefault(phase, []).append(
                duration_seconds,
            )
        self._emit(
            replace(
                self._event(
                    EvalProgressEventKind.PHASE_COMPLETED,
                    case=case,
                    repetition=repetition,
                    phase=phase,
                ),
                phase_duration_seconds=duration_seconds,
            ),
        )

    def infrastructure_retry(
        self,
        case: EvalCase,
        repetition: int,
        retry_number: int,
        total_retries: int,
    ) -> None:
        """Report a bounded infrastructure retry for a case attempt."""
        self._emit(
            replace(
                self._event(
                    EvalProgressEventKind.INFRASTRUCTURE_RETRY,
                    case=case,
                    repetition=repetition,
                    phase=EvalPhase.CANDIDATE,
                ),
                infrastructure_retry=retry_number,
                total_infrastructure_retries=total_retries,
            ),
        )

    def case_completed(
        self,
        case: EvalCase,
        repetition: int,
        duration_seconds: float,
    ) -> None:
        """Record and report a completed case repetition."""
        self.completed_cases += 1
        self._emit(
            replace(
                self._event(
                    EvalProgressEventKind.CASE_COMPLETED,
                    case=case,
                    repetition=repetition,
                ),
                case_duration_seconds=duration_seconds,
            ),
        )

    def run_finished(self) -> float:
        """Report evaluation completion and return total duration."""
        duration_seconds = self.elapsed_since(self.run_started_at)
        self._report(EvalProgressEventKind.RUN_FINISHED)
        return duration_seconds

    def run_cancelled(self) -> None:
        """Report evaluation cancellation."""
        self._report(EvalProgressEventKind.RUN_CANCELLED)

    def _report(
        self,
        kind: EvalProgressEventKind,
        case: EvalCase | None = None,
        repetition: int | None = None,
        phase: EvalPhase | None = None,
    ) -> None:
        self._emit(self._event(kind, case, repetition, phase))

    def _event(
        self,
        kind: EvalProgressEventKind,
        case: EvalCase | None = None,
        repetition: int | None = None,
        phase: EvalPhase | None = None,
    ) -> EvalProgressEvent:
        return EvalProgressEvent(
            kind=kind,
            suite_id=self.suite.id,
            completed_cases=self.completed_cases,
            total_cases=self.total_cases,
            elapsed_seconds=self.elapsed_since(self.run_started_at),
            case_id=None if case is None else case.id,
            phase=phase,
            repetition=repetition,
            total_repetitions=self.config.repetitions,
            judge_repetition=_judge_repetition(phase),
            total_judge_repetitions=_total_judge_repetitions(
                phase,
                self.config,
            ),
            estimated_remaining_seconds=self._estimated_remaining(),
        )

    def _emit(self, event: EvalProgressEvent) -> None:
        if self.reporter is None:
            return
        self.reporter.report(event)

    def _estimated_remaining(self) -> float | None:
        average_case_seconds = self._average_case_seconds()
        if average_case_seconds is None:
            return None
        return max(0, self.total_cases - self.completed_cases) * (
            average_case_seconds
        )

    def _average_case_seconds(self) -> float | None:
        required_phases = [
            EvalPhase.CANDIDATE,
            EvalPhase.DETERMINISTIC_GRADING,
        ]
        if self.config.judge_model is not None:
            required_phases.append(EvalPhase.SEMANTIC_JUDGING)
        averages = [
            _average(self.phase_durations.get(phase, []))
            for phase in required_phases
        ]
        if any(value is None for value in averages):
            return None
        return sum(value for value in averages if value is not None)


def _judge_repetition(phase: EvalPhase | None) -> int | None:
    if phase is EvalPhase.SEMANTIC_JUDGING:
        return 1
    return None


def _total_judge_repetitions(
    phase: EvalPhase | None,
    config: EvalRunConfig,
) -> int | None:
    if phase is EvalPhase.SEMANTIC_JUDGING:
        return config.judge_repetitions
    return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
