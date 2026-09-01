"""Rubric dimension domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RubricDimension:
    """One scored dimension in a strict evaluation rubric."""

    id: str
    name: str
    weight: float
    minimum_score: int
    critical: bool = False
