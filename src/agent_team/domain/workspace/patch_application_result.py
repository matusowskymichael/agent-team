"""Workspace patch application result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatchApplicationResult:
    """Result of an exact workspace text replacement."""

    path: str
    applied: bool
    before_hash: str | None
    after_hash: str | None
    line_count_delta: int
    message: str
