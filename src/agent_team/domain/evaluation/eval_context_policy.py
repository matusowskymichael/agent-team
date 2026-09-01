"""Evaluation-only context preload policy values."""

from enum import StrEnum


class EvalContextPolicy(StrEnum):
    """Feature context supplied to a candidate during evaluation."""

    STANDARD_FEATURE_CONTEXT = "standard_feature_context"
    METADATA_ONLY_FEATURE_CONTEXT = "metadata_only_feature_context"
    NO_FEATURE_PRELOAD = "no_feature_preload"
