"""Artifact kind values."""

from enum import StrEnum


class ArtifactKind(StrEnum):
    """Kind of development artifact attached to a feature."""

    REQUIREMENTS = "requirements"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION_PLAN = "implementation_plan"
    TEST_REPORT = "test_report"
    CODE_REVIEW = "code_review"
