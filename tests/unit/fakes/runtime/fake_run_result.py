"""Fake Agents SDK run result for unit tests."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FakeRunResult:
    """Fake Agents SDK run result."""

    final_output: object
    raw_responses: tuple[object, ...] = ()
