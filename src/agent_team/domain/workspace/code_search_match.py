"""Code search match result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CodeSearchMatch:
    """One bounded text-search match in a workspace file."""

    path: str
    line_number: int
    line_excerpt: str
