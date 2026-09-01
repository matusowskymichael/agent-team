"""Workspace file listing result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkspaceFileListing:
    """Bounded list of visible workspace-relative file paths."""

    files: tuple[str, ...]
    truncated: bool
