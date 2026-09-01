"""Terminal evaluation progress reporter."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import TextIO

from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_event import (
    EvalProgressEvent,
)
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)
from agent_team.interfaces.cli.eval_duration_format import format_duration

BAR_WIDTH = 20
MIN_REFRESH_INTERVAL_SECONDS = 0.1
SPINNER_FRAMES = ("|", "/", "-", "\\")


@dataclass(slots=True)
class TerminalEvalProgressReporter:
    """Render evaluation progress to a terminal stream."""

    stream: TextIO
    interactive: bool | None = None
    enabled: bool = True
    refresh_interval_seconds: float = 1.0
    auto_refresh: bool = True
    monotonic: Callable[[], float] = time.monotonic
    _current_event: EvalProgressEvent | None = field(default=None, init=False)
    _event_received_at: float = field(default=0.0, init=False)
    _last_line_length: int = field(default=0, init=False)
    _spinner_index: int = field(default=0, init=False)
    _stop_event: Event = field(default_factory=Event, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _thread: Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Resolve automatic terminal behavior."""
        if self.interactive is None:
            self.interactive = self.stream.isatty()

    def report(self, event: EvalProgressEvent) -> None:
        """Render one progress event."""
        if not self.enabled:
            return
        with self._lock:
            self._current_event = event
            self._event_received_at = self.monotonic()
            if self.interactive:
                self._report_interactive(event)
            else:
                self._report_non_interactive(event)

    def refresh(self) -> None:
        """Refresh the active interactive progress line."""
        if not self.enabled or not self.interactive:
            return
        with self._lock:
            if self._current_event is not None:
                self._write_interactive_line(self._current_event)

    def close(self) -> None:
        """Stop background refresh and clear any active progress line."""
        self._stop_refresh()
        with self._lock:
            if self.enabled and self.interactive:
                self._clear_line()

    def _report_interactive(self, event: EvalProgressEvent) -> None:
        if event.kind is EvalProgressEventKind.RUN_STARTED:
            self.stream.write(f"Evaluating {event.suite_id}\n")
            self._write_interactive_line(event)
            self._start_refresh()
        elif event.kind is EvalProgressEventKind.RUN_FINISHED:
            self._clear_line()
        elif event.kind is EvalProgressEventKind.RUN_CANCELLED:
            self._clear_line()
            duration = format_duration(event.elapsed_seconds)
            self.stream.write(f"Evaluation cancelled after {duration}\n")
        else:
            self._write_interactive_line(event)
            self._start_refresh()
        self.stream.flush()

    def _report_non_interactive(self, event: EvalProgressEvent) -> None:
        line = _non_interactive_line(event)
        if line is None:
            return
        self.stream.write(f"{line}\n")
        self.stream.flush()

    def _write_interactive_line(self, event: EvalProgressEvent) -> None:
        line = _interactive_line(
            event,
            elapsed_seconds=self._live_elapsed_seconds(event),
            spinner=SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)],
        )
        self._spinner_index += 1
        padding = max(0, self._last_line_length - len(line))
        self.stream.write(f"\r{line}{' ' * padding}")
        self._last_line_length = len(line)
        self.stream.flush()

    def _clear_line(self) -> None:
        if self._last_line_length:
            self.stream.write(f"\r{' ' * self._last_line_length}\r")
            self._last_line_length = 0
            self.stream.flush()

    def _live_elapsed_seconds(self, event: EvalProgressEvent) -> float:
        return event.elapsed_seconds + max(
            0.0,
            self.monotonic() - self._event_received_at,
        )

    def _start_refresh(self) -> None:
        if not self.auto_refresh or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def _stop_refresh(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join()
        self._thread = None

    def _refresh_loop(self) -> None:
        interval = max(
            MIN_REFRESH_INTERVAL_SECONDS,
            self.refresh_interval_seconds,
        )
        while not self._stop_event.wait(interval):
            self.refresh()


def _interactive_line(
    event: EvalProgressEvent,
    elapsed_seconds: float,
    spinner: str,
) -> str:
    return (
        f"{_progress_bar(event.completed_cases, event.total_cases)} "
        f"{event.completed_cases}/{event.total_cases} | "
        f"{event.case_id or '-'} | {_phase_label(event)} | "
        f"elapsed {format_duration(elapsed_seconds)} | "
        f"{_eta_label(event)} | {spinner}"
    )


def _non_interactive_line(event: EvalProgressEvent) -> str | None:
    if event.kind is EvalProgressEventKind.RUN_STARTED:
        line = f"Evaluating {event.suite_id}"
    elif event.kind is EvalProgressEventKind.PHASE_STARTED:
        line = (
            f"{event.completed_cases}/{event.total_cases} "
            f"{event.case_id or '-'} {_phase_label(event)}"
        )
    elif event.kind is EvalProgressEventKind.INFRASTRUCTURE_RETRY:
        line = f"{event.case_id or '-'} | {_infrastructure_retry_label(event)}"
    elif event.kind is EvalProgressEventKind.CASE_COMPLETED:
        duration = format_duration(event.case_duration_seconds)
        line = (
            f"Completed {event.completed_cases}/{event.total_cases} "
            f"{event.case_id or '-'} in {duration}"
        )
    elif event.kind is EvalProgressEventKind.RUN_FINISHED:
        duration = format_duration(event.elapsed_seconds)
        line = f"Finished evaluation in {duration}"
    elif event.kind is EvalProgressEventKind.RUN_CANCELLED:
        line = (
            "Evaluation cancelled after "
            f"{format_duration(event.elapsed_seconds)}"
        )
    else:
        line = None
    return line


def _phase_label(event: EvalProgressEvent) -> str:
    if event.kind is EvalProgressEventKind.INFRASTRUCTURE_RETRY:
        return _infrastructure_retry_label(event)
    phase = event.phase
    if phase is None:
        return "starting"
    repetition = _repetition_label(event)
    if phase is EvalPhase.SEMANTIC_JUDGING:
        return f"{phase.value} {repetition} {_judge_label(event)}"
    return f"{phase.value} {repetition}"


def _repetition_label(event: EvalProgressEvent) -> str:
    if event.repetition is None:
        return ""
    return f"{event.repetition}/{event.total_repetitions}"


def _judge_label(event: EvalProgressEvent) -> str:
    if event.judge_repetition is None or event.total_judge_repetitions is None:
        return ""
    return f"judge {event.judge_repetition}/{event.total_judge_repetitions}"


def _infrastructure_retry_label(event: EvalProgressEvent) -> str:
    retry = event.infrastructure_retry
    total = event.total_infrastructure_retries
    if retry is None or total is None:
        return "infrastructure retry"
    return f"infrastructure retry {retry}/{total}"


def _eta_label(event: EvalProgressEvent) -> str:
    if event.estimated_remaining_seconds is None:
        return "ETA calculating"
    return f"ETA ~{format_duration(event.estimated_remaining_seconds)}"


def _progress_bar(completed_cases: int, total_cases: int) -> str:
    if total_cases <= 0:
        filled = 0
    else:
        filled = round((completed_cases / total_cases) * BAR_WIDTH)
    filled = max(0, min(BAR_WIDTH, filled))
    return f"[{'#' * filled}{'-' * (BAR_WIDTH - filled)}]"
