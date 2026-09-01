"""Deterministic grade domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeterministicGrade:
    """Result of objective deterministic evaluation checks."""

    passed: bool
    hard_gate_failed: bool
    reasons: tuple[str, ...]
