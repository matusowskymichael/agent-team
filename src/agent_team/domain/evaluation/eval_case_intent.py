"""Evaluation case intent values."""

from enum import StrEnum


class EvalCaseIntent(StrEnum):
    """Focused purpose of one evaluation case."""

    UNSPECIFIED = "unspecified"
    OUTCOME_GROUNDING = "outcome_grounding"
    TOOL_DISPATCH = "tool_dispatch"
    AUTHORIZED_MUTATION = "authorized_mutation"
    CAPABILITY_BOUNDARY = "capability_boundary"
    SESSION_BOUNDARY = "session_boundary"
