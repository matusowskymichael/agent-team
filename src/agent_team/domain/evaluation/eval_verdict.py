"""Evaluation verdict values."""

from enum import StrEnum


class EvalVerdict(StrEnum):
    """Verdict values used by deterministic graders and judges."""

    PASS = "pass"  # noqa: S105 - Evaluation verdict, not a password.
    FAIL = "fail"
    NOT_RUN = "not_run"
    PASSED = "passed"
    DETERMINISTIC_FAILED = "deterministic_failed"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    NOT_JUDGED = "not_judged"
    JUDGE_FAILED = "judge_failed"
    JUDGE_ERROR = "judge_error"
    AMBIGUOUS = "ambiguous"
