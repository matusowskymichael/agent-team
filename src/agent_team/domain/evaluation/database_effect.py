"""Observed database effect domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatabaseEffect:
    """A persisted database change observed during evaluation."""

    table: str
    operation: str
    field_values: dict[str, object]
