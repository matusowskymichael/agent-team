"""Tool invocation audit status values."""

from enum import StrEnum


class ToolInvocationStatus(StrEnum):
    """Lifecycle status for an audited tool invocation."""

    ALLOWED = "allowed"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
