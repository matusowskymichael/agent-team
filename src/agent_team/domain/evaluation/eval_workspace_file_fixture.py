"""Evaluation workspace file fixture."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalWorkspaceFileFixture:
    """One file to seed into an isolated evaluation workspace."""

    path: str
    content: str
