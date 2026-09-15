"""Use case for deterministic task verification."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_team.application.workflow.task_verifier import TaskVerifier
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.development_task_not_found_error import (
    DevelopmentTaskNotFoundError,
)
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_submission_error import (
    TaskSubmissionError,
)
from agent_team.domain.workflow.task_transition_error import (
    TaskTransitionError,
)
from agent_team.domain.workflow.task_verification_evidence import (
    TaskVerificationEvidence,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_profiles import (
    is_valid_verification_contract,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)
from agent_team.domain.workflow.workflow_repository import WorkflowRepository

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


@dataclass(frozen=True, slots=True)
class TaskVerificationService:
    """Verify submitted tasks and persist authoritative outcomes."""

    repository: WorkflowRepository
    verifier: TaskVerifier

    def verify_task(
        self,
        task_id: int,
        workspace_root: Path,
    ) -> TaskVerificationEvidence:
        """Verify a task waiting in verification_pending state."""
        task = self._require_task(task_id)
        handoff = self._require_handoff(task_id)
        existing = self.repository.latest_task_verification(task_id)
        if existing is not None and existing.submission_id == handoff.id:
            return existing
        if task.status is not TaskStatus.VERIFICATION_PENDING:
            raise TaskTransitionError(
                "Task is not awaiting deterministic verification.",
            )
        result = self._run_verifier(task, handoff, workspace_root)
        next_status = _next_status(result.outcome)
        return self.repository.record_task_verification(
            task_id=task_id,
            submission_id=handoff.id,
            result=result,
            next_status=next_status,
        )

    def verify_task_if_pending(
        self,
        task_id: int,
        workspace_root: Path,
    ) -> TaskVerificationEvidence | None:
        """Verify a task only when it is awaiting verification."""
        task = self._require_task(task_id)
        if task.status is not TaskStatus.VERIFICATION_PENDING:
            return None
        return self.verify_task(task_id, workspace_root)

    def _require_task(self, task_id: int) -> DevelopmentTask:
        task = self.repository.get_task(task_id)
        if task is None:
            raise DevelopmentTaskNotFoundError(
                f"Development task {task_id} was not found.",
            )
        return task

    def _require_handoff(self, task_id: int) -> TaskHandoff:
        handoff = self.repository.latest_task_handoff(task_id)
        if handoff is None:
            raise TaskSubmissionError(
                "Task has no persisted handoff to verify.",
            )
        return handoff

    def _run_verifier(
        self,
        task: DevelopmentTask,
        handoff: TaskHandoff,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        if not is_valid_verification_contract(
            task.assigned_role,
            task.verification_contract,
        ):
            return _blocked_result(
                "Task verification configuration is missing or invalid.",
            )
        return self.verifier.verify(task, handoff, workspace_root)


def _next_status(outcome: TaskVerificationOutcome) -> TaskStatus:
    if outcome is TaskVerificationOutcome.PASSED:
        return TaskStatus.COMPLETED
    if outcome is TaskVerificationOutcome.FAILED:
        return TaskStatus.IN_PROGRESS
    return TaskStatus.BLOCKED


def _blocked_result(feedback: str) -> TaskVerificationResult:
    timestamp = datetime.now(UTC)
    return TaskVerificationResult(
        verifier_name="configuration",
        outcome=TaskVerificationOutcome.BLOCKED,
        failure_classification=FailureClassification.CONFIGURATION_ERROR,
        feedback=feedback,
        checks=(),
        started_at=timestamp,
        ended_at=timestamp,
    )
