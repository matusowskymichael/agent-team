"""Observed tool call domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObservedToolCall:
    """A tool call observed during a candidate agent run."""

    name: str
    arguments: dict[str, object]
    status: str
    reached_mcp: bool = True
