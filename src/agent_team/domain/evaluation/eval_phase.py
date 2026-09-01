"""Evaluation progress phase values."""

from enum import StrEnum


class EvalPhase(StrEnum):
    """Major phases for one evaluation case repetition."""

    CANDIDATE = "candidate"
    DETERMINISTIC_GRADING = "deterministic grading"
    SEMANTIC_JUDGING = "semantic judging"
