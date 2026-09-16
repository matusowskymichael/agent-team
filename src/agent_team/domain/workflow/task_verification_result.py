"""Task verification result model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.task_verification_check_result import (
    TaskVerificationCheckResult,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


@dataclass(frozen=True, slots=True)
class TaskVerificationResult:
    """Deterministic verification result before persistence."""

    verifier_name: str
    outcome: TaskVerificationOutcome
    failure_classification: FailureClassification
    feedback: str
    checks: tuple[TaskVerificationCheckResult, ...]
    started_at: datetime
    ended_at: datetime
