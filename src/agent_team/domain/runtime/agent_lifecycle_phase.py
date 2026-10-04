"""Durable developer task lifecycle phases."""

from enum import StrEnum


class AgentLifecyclePhase(StrEnum):
    """Typed lifecycle milestones derived from trusted state and effects."""

    STARTED = "started"
    INSPECTED = "inspected"
    ACTIVATED = "activated"
    IMPLEMENTED = "implemented"
    CHECKED = "checked"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    COMPLETED = "completed"
    BLOCKED = "blocked"
