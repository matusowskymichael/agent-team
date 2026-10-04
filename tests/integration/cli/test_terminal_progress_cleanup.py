"""Stop terminal heartbeat threads on every evaluation exit path."""

import threading
from dataclasses import replace
from io import StringIO

import pytest

from agent_team.domain.evaluation.eval_progress_event import EvalProgressEvent
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)
from agent_team.interfaces.cli.terminal_eval_progress_reporter import (
    TerminalEvalProgressReporter,
)


class TestTerminalProgressCleanup:
    """Ensure refresh workers never outlive terminal completion events."""

    @pytest.mark.parametrize(
        "terminal_kind",
        (
            EvalProgressEventKind.RUN_FINISHED,
            EvalProgressEventKind.RUN_CANCELLED,
        ),
    )
    def test_heartbeat_thread_stops_at_terminal_event(
        self,
        terminal_progress_event: EvalProgressEvent,
        terminal_kind: EvalProgressEventKind,
    ) -> None:
        """Join refresh without waiting for its scheduled heartbeat."""
        stream = StringIO()
        reporter = TerminalEvalProgressReporter(
            stream=stream, interactive=True, refresh_interval_seconds=30
        )
        existing = set(threading.enumerate())
        try:
            reporter.report(terminal_progress_event)
            workers = set(threading.enumerate()) - existing
            assert len(workers) == 1
            reporter.report(
                replace(terminal_progress_event, kind=terminal_kind)
            )
            assert all(not worker.is_alive() for worker in workers)
            finished = stream.getvalue()
            reporter.refresh()
            assert stream.getvalue() == finished
        finally:
            reporter.close()

    def test_explicit_close_stops_active_heartbeat(
        self, terminal_progress_event: EvalProgressEvent
    ) -> None:
        """Cleanup closes an active reporter after infrastructure failure."""
        reporter = TerminalEvalProgressReporter(
            stream=StringIO(), interactive=True, refresh_interval_seconds=30
        )
        existing = set(threading.enumerate())
        reporter.report(terminal_progress_event)
        workers = set(threading.enumerate()) - existing
        assert len(workers) == 1
        reporter.close()
        assert all(not worker.is_alive() for worker in workers)
