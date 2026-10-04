"""Shared monotonic deadline for one cancellation cleanup episode."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)


@dataclass(slots=True)
class AgentCleanupBudget:
    """Share bounded cleanup time across adapter and harness finalization."""

    grace_seconds: float = 10.0
    clock: Callable[[], float] = time.monotonic
    _deadline: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Use the same finite, ten-second validation as watchdog settings."""
        AgentWatchdogSettings(cleanup_grace_seconds=self.grace_seconds)

    def begin(self) -> float:
        """Start once so nested cleanup cannot restart its grace period."""
        if self._deadline is None:
            self._deadline = self.clock() + self.grace_seconds
        return self._deadline

    def remaining(self) -> float:
        """Start if necessary and return the unspent cleanup time."""
        return max(0.0, self.begin() - self.clock())

    def reset(self) -> None:
        """Finish successful cleanup before a new normal work episode."""
        self._deadline = None
