"""Task verification evidence model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.task_verification_check import (
    TaskVerificationCheck,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


@dataclass(frozen=True, slots=True)
class TaskVerificationEvidence:
    """Persisted deterministic evidence for a submitted task."""

    id: int
    task_id: int
    submission_id: int
    verifier_name: str
    outcome: TaskVerificationOutcome
    failure_classification: FailureClassification
    feedback: str
    checks: tuple[TaskVerificationCheck, ...]
    started_at: datetime
    ended_at: datetime
