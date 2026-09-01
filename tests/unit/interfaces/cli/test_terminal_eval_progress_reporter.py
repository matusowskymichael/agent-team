"""Tests for terminal evaluation progress rendering."""

from dataclasses import replace
from io import StringIO

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
