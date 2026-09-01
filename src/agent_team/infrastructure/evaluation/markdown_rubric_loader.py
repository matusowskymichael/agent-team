"""Strict Markdown rubric loader."""

from dataclasses import dataclass
from pathlib import Path

from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.evaluation.rubric_dimension import RubricDimension
from agent_team.infrastructure.evaluation.eval_hashes import hash_file

DIMENSION_PREFIX = "- "
EXPECTED_DIMENSION_FIELDS = 5


@dataclass(frozen=True, slots=True)
class MarkdownRubricLoader:
    """Parse a strict Markdown rubric without external dependencies."""

    def load(self, path: Path) -> Rubric:
        """Load a fail-closed Markdown rubric."""
        text = path.read_text()
        values = _metadata(text)
        dimensions = _dimensions(text)
        if not dimensions:
            raise ValueError("Rubric must define at least one dimension.")
        return Rubric(
            id=values["rubric_id"],
            version=values["version"],
            threshold=float(values["threshold"]),
            dimensions=dimensions,
            content_hash=hash_file(path),
            source_text=text,
        )


def _metadata(text: str) -> dict[str, str]:
    required_keys = {"rubric_id", "version", "threshold"}
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        clean_key = key.strip()
        if clean_key in required_keys:
            values[clean_key] = value.strip()
    missing_keys = required_keys - set(values)
    if missing_keys:
        raise ValueError(
            f"Rubric is missing metadata: {', '.join(sorted(missing_keys))}.",
        )
    return values


def _dimensions(text: str) -> tuple[RubricDimension, ...]:
    dimensions: list[RubricDimension] = []
    in_dimensions = False
    for line in text.splitlines():
        if line.strip() == "## Dimensions":
            in_dimensions = True
            continue
        if in_dimensions and line.startswith("## "):
            break
        if in_dimensions and line.startswith(DIMENSION_PREFIX):
            dimensions.append(_dimension(line.removeprefix(DIMENSION_PREFIX)))
    return tuple(dimensions)


def _dimension(line: str) -> RubricDimension:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != EXPECTED_DIMENSION_FIELDS:
        raise ValueError("Rubric dimension rows must contain five fields.")
    dimension_id, name, weight, critical, minimum_score = parts
    return RubricDimension(
        id=dimension_id,
        name=name,
        weight=float(weight),
        critical=critical == "critical",
        minimum_score=int(minimum_score),
    )
