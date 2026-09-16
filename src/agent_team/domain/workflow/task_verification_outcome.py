"""Task verification outcome values."""

from enum import StrEnum


class TaskVerificationOutcome(StrEnum):
    """Final deterministic verification outcome."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
