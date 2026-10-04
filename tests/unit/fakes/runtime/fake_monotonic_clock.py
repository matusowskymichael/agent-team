"""Controllable monotonic runtime clock."""

from dataclasses import dataclass


@dataclass(slots=True)
class FakeMonotonicClock:
    """Advance watchdog time without real waiting."""

    now: float = 0.0

    def __call__(self) -> float:
        """Return the deterministic monotonic timestamp."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the test clock by a deterministic duration."""
        self.now += seconds
