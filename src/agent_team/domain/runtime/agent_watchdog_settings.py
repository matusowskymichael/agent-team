"""Finite operation and convergence safety watchdog settings."""

import math
from dataclasses import dataclass

MAX_CLEANUP_GRACE_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class AgentWatchdogSettings:
    """Bound hanging operations and stagnation without a total turn cap."""

    provider_response_timeout_seconds: float = 900.0
    segment_timeout_seconds: float = 1800.0
    no_advancement_timeout_seconds: float = 1800.0
    stagnant_segment_threshold: int = 4
    equivalent_failure_threshold: int = 3
    cleanup_grace_seconds: float = 10.0

    def __post_init__(self) -> None:
        """Require finite watchdogs and bounded cancellation cleanup."""
        for name in (
            "provider_response_timeout_seconds",
            "segment_timeout_seconds",
            "no_advancement_timeout_seconds",
            "cleanup_grace_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite.")
        for name in (
            "stagnant_segment_threshold",
            "equivalent_failure_threshold",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.cleanup_grace_seconds > MAX_CLEANUP_GRACE_SECONDS:
            raise ValueError("cleanup_grace_seconds must not exceed 10.")
