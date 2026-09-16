"""Deterministic verifier fake exercising real verification persistence."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent_team.application.audit.audit_sanitizer import hash_text
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_check_result import (
    TaskVerificationCheckResult,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


@dataclass(slots=True)
class CompletionVerifier:
    """Pass only after the workspace reaches the configured repair revision."""

    required_revision: int = 1
    submissions: list[TaskHandoff] = field(default_factory=list[TaskHandoff])

    def verify(
        self,
        task: DevelopmentTask,
        handoff: TaskHandoff,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        """Judge persisted candidate contents with no model execution."""
        assert task.status is TaskStatus.VERIFICATION_PENDING
        self.submissions.append(handoff)
        source = (workspace_root / "backend/auth.py").read_text(
            encoding="utf-8"
        )
        passed = f"return {self.required_revision}\n" in source
        timestamp = datetime.now(UTC)
        feedback = (
            "Required revision verified."
            if passed
            else f"Repair to revision {self.required_revision}; "
            "token=fixture-private-verification-value"
        )
        return TaskVerificationResult(
            verifier_name="deterministic-completion-fixture",
            outcome=(
                TaskVerificationOutcome.PASSED
                if passed
                else TaskVerificationOutcome.FAILED
            ),
            failure_classification=(
                FailureClassification.NONE
                if passed
                else FailureClassification.CHECK_FAILED
            ),
            feedback=feedback,
            checks=(
                TaskVerificationCheckResult(
                    name="backend:behavior",
                    started_at=timestamp,
                    ended_at=timestamp,
                    exit_code=0 if passed else 1,
                    timed_out=False,
                    stdout_hash=hash_text(""),
                    stdout_excerpt="",
                    stderr_hash=hash_text(""),
                    stderr_excerpt="",
                ),
            ),
            started_at=timestamp,
            ended_at=timestamp,
        )
