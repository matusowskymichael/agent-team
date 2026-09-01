"""Task status values."""

from enum import StrEnum


class TaskStatus(StrEnum):
    """Lifecycle status for a development task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
