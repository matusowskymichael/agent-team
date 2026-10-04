"""Tests for terminal evaluation progress rendering."""

from dataclasses import replace
from io import StringIO

import pytest

from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_event import (
    EvalProgressEvent,
)
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)
from agent_team.interfaces.cli.eval_duration_format import format_duration
from agent_team.interfaces.cli.terminal_eval_progress_reporter import (
    TerminalEvalProgressReporter,
)


class _Stream(StringIO):
    def __init__(self, interactive: bool) -> None:
        super().__init__()
        self.interactive = interactive

    def isatty(self) -> bool:
        return self.interactive


class _ManualClock:
    def __init__(self) -> None:
        self.current = 0.0

    def monotonic(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


class TestTerminalEvalProgressReporter:
    """Terminal progress reporter behavior tests."""

    @pytest.mark.parametrize(
        ("classification", "expected"),
        (
            ("check_failed", "check_failed"),
            ("timeout", "timeout"),
            ("/private/source.py:secret", "-"),
        ),
    )
    def test_heartbeat_exposes_only_safe_verification_classifications(
        self,
        liveness_heartbeat: EvalProgressEvent,
        classification: str,
        expected: str,
    ) -> None:
        """Explain verification failures without arbitrary diagnostic text."""
        snapshot = liveness_heartbeat.liveness_snapshot
        assert snapshot is not None
        event = replace(
            liveness_heartbeat,
            liveness_snapshot=replace(
                snapshot,
                last_verification_outcome="failed",
                verification_failure_classification=classification,
            ),
        )
        stream = _Stream(interactive=False)
        reporter = TerminalEvalProgressReporter(
            stream=stream, interactive=False, auto_refresh=False
        )
        reporter.report(event)
        output = stream.getvalue()
        assert f"verification failed ({expected})" in output
        assert "/private" not in output

    def test_heartbeat_metadata_renders_live_advancement_and_deadline(
        self, liveness_heartbeat: EvalProgressEvent
    ) -> None:
        """Expose bounded runtime facts as inference continues waiting."""
        clock = _ManualClock()
        stream = _Stream(interactive=True)
        reporter = TerminalEvalProgressReporter(
            stream=stream,
            interactive=True,
            auto_refresh=False,
            monotonic=clock.monotonic,
        )
        reporter.report(liveness_heartbeat)
        clock.advance(30)
        reporter.refresh()
        output = stream.getvalue()
        for value in (
            "fd-dev-002",
            "segment 2",
            "task in_progress",
            "phase checked",
            "waiting model",
            "last operation run_check",
            "check failed",
            "last advancement 00:04:40 ago",
            "case deadline 00:45:00",
            "remaining 00:32:00",
        ):
            assert value in output

    def test_heartbeat_metadata_never_prints_private_fields(
        self, liveness_heartbeat: EvalProgressEvent
    ) -> None:
        """Unknown metadata cannot inject paths, arguments or secret output."""
        snapshot = liveness_heartbeat.liveness_snapshot
        assert snapshot is not None
        event = replace(
            liveness_heartbeat,
            liveness_snapshot=replace(
                snapshot,
                last_tool_name="read_file('/private/source.py')",
                task_status="secret-token",
                waiting_phase="private prompt",
                last_check_outcome="source body",
                changed_paths=("/private",),
            ),
        )
        stream = _Stream(interactive=False)
        reporter = TerminalEvalProgressReporter(
            stream=stream, interactive=False, auto_refresh=False
        )
        reporter.report(event)
        output = stream.getvalue()
        assert "segment 2" in output
        for private in (
            "/private",
            "secret-token",
            "private prompt",
            "source body",
            "read_file(",
        ):
            assert private not in output

    def test_finished_reporter_does_not_resume_refreshing(self) -> None:
        """Terminal events stop further refreshes of completed candidates."""
        stream = _Stream(interactive=True)
        reporter = TerminalEvalProgressReporter(
            stream=stream, interactive=True, auto_refresh=False
        )
        reporter.report(_event(EvalProgressEventKind.PHASE_STARTED))
        reporter.report(_event(EvalProgressEventKind.RUN_FINISHED))
        finished = stream.getvalue()
        reporter.refresh()
        assert stream.getvalue() == finished

    def test_interactive_rendering_refreshes_elapsed_time(self) -> None:
        """Render an updating interactive line without ANSI escape codes."""
        clock = _ManualClock()
        stream = _Stream(interactive=True)
        reporter = TerminalEvalProgressReporter(
            stream=stream,
            interactive=True,
            auto_refresh=False,
            monotonic=clock.monotonic,
        )

        reporter.report(_event(EvalProgressEventKind.RUN_STARTED))
        reporter.report(
            _event(
                EvalProgressEventKind.PHASE_STARTED,
                case_id="ba-dev-006",
                phase=EvalPhase.CANDIDATE,
            ),
        )
        clock.advance(3661)
        reporter.refresh()
        reporter.close()

        output = stream.getvalue()
        assert "\r" in output
        assert "\x1b" not in output
        assert "ba-dev-006" in output
        assert "candidate 1/1" in output
        assert "elapsed 01:01:01" in output
        assert "ETA calculating" in output

    def test_non_tty_rendering_uses_plain_milestones(self) -> None:
        """Render non-interactive progress without animation or ANSI."""
        stream = _Stream(interactive=False)
        reporter = TerminalEvalProgressReporter(
            stream=stream,
            interactive=False,
            auto_refresh=False,
        )

        reporter.report(_event(EvalProgressEventKind.RUN_STARTED))
        reporter.report(
            _event(
                EvalProgressEventKind.PHASE_STARTED,
                case_id="ba-dev-001",
                phase=EvalPhase.DETERMINISTIC_GRADING,
            ),
        )
        reporter.report(
            replace(
                _event(
                    EvalProgressEventKind.CASE_COMPLETED,
                    case_id="ba-dev-001",
                    completed_cases=1,
                ),
                case_duration_seconds=65,
            ),
        )
        reporter.report(
            replace(
                _event(
                    EvalProgressEventKind.RUN_FINISHED,
                    completed_cases=1,
                ),
                completed_cases=1,
                elapsed_seconds=65,
            ),
        )

        output = stream.getvalue()
        assert "\r" not in output
        assert "\x1b" not in output
        assert "Evaluating business_analyst_development" in output
        assert "0/1 ba-dev-001 deterministic grading 1/1" in output
        assert "Completed 1/1 ba-dev-001 in 00:01:05" in output
        assert "Finished evaluation in 00:01:05" in output

    def test_eta_renders_when_available(self) -> None:
        """Render approximate ETA after samples are available."""
        stream = _Stream(interactive=True)
        reporter = TerminalEvalProgressReporter(
            stream=stream,
            interactive=True,
            auto_refresh=False,
        )

        reporter.report(
            replace(
                _event(
                    EvalProgressEventKind.PHASE_STARTED,
                    case_id="ba-dev-002",
                    phase=EvalPhase.SEMANTIC_JUDGING,
                ),
                estimated_remaining_seconds=125,
                judge_repetition=1,
                total_judge_repetitions=2,
            ),
        )

        output = stream.getvalue()
        assert "semantic judging 1/1 judge 1/2" in output
        assert "ETA ~00:02:05" in output

    def test_retry_progress_renders_concise_line(self) -> None:
        """Render infrastructure retries as distinct progress events."""
        stream = _Stream(interactive=False)
        reporter = TerminalEvalProgressReporter(
            stream=stream,
            interactive=False,
            auto_refresh=False,
        )

        reporter.report(
            replace(
                _event(
                    EvalProgressEventKind.INFRASTRUCTURE_RETRY,
                    case_id="ba-dev-012",
                    phase=EvalPhase.CANDIDATE,
                ),
                infrastructure_retry=1,
                total_infrastructure_retries=1,
            ),
        )

        assert "ba-dev-012 | infrastructure retry 1/1" in stream.getvalue()

    def test_cancellation_clears_line_and_prints_duration(self) -> None:
        """Finish the progress line on cancellation."""
        stream = _Stream(interactive=True)
        reporter = TerminalEvalProgressReporter(
            stream=stream,
            interactive=True,
            auto_refresh=False,
        )

        reporter.report(
            _event(
                EvalProgressEventKind.PHASE_STARTED,
                case_id="ba-dev-001",
                phase=EvalPhase.CANDIDATE,
            ),
        )
        reporter.report(
            replace(
                _event(EvalProgressEventKind.RUN_CANCELLED),
                elapsed_seconds=7322,
            ),
        )

        assert "Evaluation cancelled after 02:02:02" in stream.getvalue()

    def test_finish_clears_progress_before_summary_output(self) -> None:
        """Clear the interactive line before regular summary output."""
        stream = _Stream(interactive=True)
        reporter = TerminalEvalProgressReporter(
            stream=stream,
            interactive=True,
            auto_refresh=False,
        )

        reporter.report(
            _event(
                EvalProgressEventKind.PHASE_STARTED,
                case_id="ba-dev-001",
                phase=EvalPhase.CANDIDATE,
            ),
        )
        reporter.report(_event(EvalProgressEventKind.RUN_FINISHED))
        stream.write("Eval run: run-1\n")

        output = stream.getvalue()
        assert "\rEval run: run-1" in output

    def test_duration_format_handles_sub_hour_and_hour_values(self) -> None:
        """Format durations with stable HH:MM:SS output."""
        assert format_duration(None) == "-"
        assert format_duration(65.9) == "00:01:05"
        assert format_duration(3723.1) == "01:02:03"


def _event(
    kind: EvalProgressEventKind,
    case_id: str | None = None,
    phase: EvalPhase | None = None,
    completed_cases: int = 0,
) -> EvalProgressEvent:
    return EvalProgressEvent(
        kind=kind,
        suite_id="business_analyst_development",
        completed_cases=completed_cases,
        total_cases=1,
        elapsed_seconds=0,
        case_id=case_id,
        phase=phase,
        repetition=1 if case_id is not None else None,
        total_repetitions=1,
        judge_repetition=None,
        total_judge_repetitions=None,
        estimated_remaining_seconds=None,
        case_duration_seconds=None,
    )
