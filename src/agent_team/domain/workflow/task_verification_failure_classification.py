"""Task verification failure classification values."""

from enum import StrEnum


class TaskVerificationFailureClassification(StrEnum):
    """Actionable classification for a verification result."""

    NONE = "none"
    CHECK_FAILED = "check_failed"
    TIMEOUT = "timeout"
    CONFIGURATION_ERROR = "configuration_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
