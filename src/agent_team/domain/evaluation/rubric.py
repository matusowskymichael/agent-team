"""Rubric domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.rubric_dimension import RubricDimension


@dataclass(frozen=True, slots=True)
class Rubric:
    """Strict Markdown rubric parsed for human and judge evaluation."""

    id: str
    version: str
    threshold: float
    dimensions: tuple[RubricDimension, ...]
    content_hash: str
    source_text: str
