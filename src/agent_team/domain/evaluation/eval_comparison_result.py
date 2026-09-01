"""Evaluation comparison result domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalComparisonResult:
    """Pairwise comparison between two evaluation runs."""

    baseline_id: str
    candidate_id: str
    improved_cases: tuple[str, ...]
    regressed_cases: tuple[str, ...]
    unchanged_cases: tuple[str, ...]
    warnings: tuple[str, ...]
    deterministic_improved_cases: tuple[str, ...] = ()
    deterministic_regressed_cases: tuple[str, ...] = ()
    deterministic_uncomparable_cases: tuple[str, ...] = ()
    semantic_improved_cases: tuple[str, ...] = ()
    semantic_regressed_cases: tuple[str, ...] = ()
    semantic_uncomparable_cases: tuple[str, ...] = ()
