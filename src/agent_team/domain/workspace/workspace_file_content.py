"""Workspace file content result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkspaceFileContent:
    """Bounded content read from a workspace file."""

    path: str
    content: str
    content_hash: str
    truncated: bool
