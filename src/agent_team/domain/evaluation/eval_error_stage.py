"""Evaluation error stage values."""

from enum import StrEnum


class EvalErrorStage(StrEnum):
    """Stage at which an expected evaluation error occurs."""

    CANDIDATE_EXECUTION = "candidate_execution"
    INFRASTRUCTURE_SETUP = "infrastructure_setup"
    SESSION_BINDING = "session_binding"
