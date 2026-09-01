"""Workspace check run result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckRunResult:
    """Bounded result from an allowlisted workspace check."""

    name: str
    exit_code: int
    stdout_excerpt: str
    stderr_excerpt: str
    timed_out: bool
