"""Task verification check result model."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TaskVerificationCheckResult:
    """Bounded deterministic result from one trusted verification check."""

    name: str
    started_at: datetime
    ended_at: datetime
    exit_code: int
    timed_out: bool
    stdout_hash: str
    stdout_excerpt: str
    stderr_hash: str
    stderr_excerpt: str
