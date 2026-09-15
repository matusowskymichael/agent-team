"""Agent implementation status values."""

from enum import StrEnum


class AgentImplementationStatus(StrEnum):
    """Whether a role has a runnable implementation."""

    RUNNABLE = "runnable"
    MANUAL_COORDINATOR = "manual_coordinator"
    PLACEHOLDER = "placeholder"
