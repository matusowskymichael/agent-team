"""Evaluation progress reporting port."""

from typing import Protocol

from agent_team.domain.evaluation.eval_progress_event import (
    EvalProgressEvent,
)


class EvalProgressReporter(Protocol):
    """Port for receiving evaluation progress events."""

    def report(self, event: EvalProgressEvent) -> None:
        """Record or display one evaluation progress event."""
        ...
