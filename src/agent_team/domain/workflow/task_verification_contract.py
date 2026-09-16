"""Task verification contract model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskVerificationContract:
    """Machine-readable trusted checks required for task completion."""

    profile_name: str
    required_checks: tuple[str, ...]
