"""Monotonic evaluation clock port."""

from typing import Protocol


class EvalClock(Protocol):
    """Port for monotonic evaluation timing."""

    def monotonic(self) -> float:
        """Return a monotonic timestamp in seconds."""
        ...
