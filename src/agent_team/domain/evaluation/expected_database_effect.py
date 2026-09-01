"""Expected database effect domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedDatabaseEffect:
    """A database effect expected or forbidden by an evaluation case."""

    table: str
    operation: str
    field_values: dict[str, object]
