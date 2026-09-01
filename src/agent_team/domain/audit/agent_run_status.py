"""Agent run audit status values."""

from enum import StrEnum


class AgentRunStatus(StrEnum):
    """Lifecycle status for an audited agent run."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
