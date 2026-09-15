"""Persisted task verification check model."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TaskVerificationCheck:
    """Persisted bounded evidence for one verification check."""

    id: int
    verification_id: int
    name: str
    started_at: datetime
    ended_at: datetime
    exit_code: int
    timed_out: bool
    stdout_hash: str
    stdout_excerpt: str
    stderr_hash: str
    stderr_excerpt: str
