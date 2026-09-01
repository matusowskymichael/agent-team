"""Feature status values."""

from enum import StrEnum


class FeatureStatus(StrEnum):
    """Lifecycle status for a development feature."""

    DRAFT = "draft"
    ANALYSIS = "analysis"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    COMPLETED = "completed"
