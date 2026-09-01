"""Tool capability classification values."""

from enum import StrEnum


class ToolClassification(StrEnum):
    """Security classification for a tool invocation."""

    READ_ONLY = "read_only"
    MUTATING = "mutating"
    PRIVILEGED = "privileged"
    DESTRUCTIVE = "destructive"
    PROHIBITED = "prohibited"
