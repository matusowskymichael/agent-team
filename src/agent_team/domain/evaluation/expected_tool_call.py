"""Expected tool call domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedToolCall:
    """A tool call expected in an evaluation case."""

    name: str
    arguments_subset: dict[str, object]
    order: int | None = None
