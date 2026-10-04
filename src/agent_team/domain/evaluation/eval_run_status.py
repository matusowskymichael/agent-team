"""Durable lifecycle states for evaluation checkpoints."""

from enum import StrEnum


class EvalRunStatus(StrEnum):
    """Distinguish active, complete, interrupted, and failed evaluations."""

    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
