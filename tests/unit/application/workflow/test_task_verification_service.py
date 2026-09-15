"""Tests for deterministic task verification service."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_team.application.sessions.workspace_identity import (
    workspace_identity_hash,
)
from agent_team.application.workflow.task_verification_service import (
    TaskVerificationService,
)
from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_submission_error import (
    TaskSubmissionError,
)
from agent_team.domain.workflow.task_transition_error import (
    TaskTransitionError,
)
from agent_team.domain.workflow.task_verification_check_result import (
    TaskVerificationCheckResult,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)
from agent_team.domain.workflow.task_verification_workspace_error import (
    TaskVerificationWorkspaceError,
)
from tests.unit.fakes.audit.audit_record_factories import (
    make_agent_run_record,
)
from tests.unit.fakes.audit.fake_agent_audit_reader import (
    FakeAgentAuditReader,
)
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


@dataclass(slots=True)
class _FakeVerifier:
    result: TaskVerificationResult
    calls: int = 0

    def verify(
        self,
        task: object,
        handoff: object,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        """Return the configured verification result."""
        _ = (task, handoff, workspace_root)
        self.calls += 1
        return self.result


@dataclass(slots=True)
class _StatusChangingVerifier:
    repository: FakeWorkflowRepository
    result: TaskVerificationResult
    calls: int = 0

    def verify(
        self,
        task: DevelopmentTask,
        handoff: TaskHandoff,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        """Change task status before returning verification output."""
        _ = (handoff, workspace_root)
        self.calls += 1
        self.repository.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        return self.result


FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


class TestTaskVerificationService:
    """TaskVerificationService behavior tests."""

    def test_successful_verification_marks_task_completed(self) -> None:
        """Persist passed evidence and complete the task."""
        repository, task_id = _submitted_task()
        verifier = _FakeVerifier(
            _verification_result(TaskVerificationOutcome.PASSED),
        )
        service = TaskVerificationService(repository, verifier)

        evidence = service.verify_task(task_id, Path("workspace"))
        completed_task = repository.get_task(task_id)

        assert evidence.outcome is TaskVerificationOutcome.PASSED
        assert completed_task is not None
        assert completed_task.status is TaskStatus.COMPLETED
        assert verifier.calls == 1

    def test_failed_verification_returns_task_to_in_progress(self) -> None:
        """Persist failed evidence and reopen the task."""
        repository, task_id = _submitted_task()
        verifier = _FakeVerifier(
            _verification_result(
                TaskVerificationOutcome.FAILED,
                FailureClassification.CHECK_FAILED,
            ),
        )
        service = TaskVerificationService(repository, verifier)

        evidence = service.verify_task(task_id, Path("workspace"))
        reopened_task = repository.get_task(task_id)

        assert evidence.failure_classification is (
            FailureClassification.CHECK_FAILED
        )
        assert reopened_task is not None
        assert reopened_task.status is TaskStatus.IN_PROGRESS

    def test_blocked_verification_marks_task_blocked(self) -> None:
        """Persist blocked evidence when verification cannot run."""
        repository, task_id = _submitted_task()
        verifier = _FakeVerifier(
            _verification_result(
                TaskVerificationOutcome.BLOCKED,
                FailureClassification.INFRASTRUCTURE_ERROR,
            ),
        )
        service = TaskVerificationService(repository, verifier)

        evidence = service.verify_task(task_id, Path("workspace"))
        blocked_task = repository.get_task(task_id)

        assert evidence.outcome is TaskVerificationOutcome.BLOCKED
        assert blocked_task is not None
        assert blocked_task.status is TaskStatus.BLOCKED

    def test_resume_is_idempotent_for_same_submission(self) -> None:
        """Return existing evidence for a previously verified handoff."""
        repository, task_id = _submitted_task()
        verifier = _FakeVerifier(
            _verification_result(TaskVerificationOutcome.PASSED),
        )
        service = TaskVerificationService(repository, verifier)
        first_evidence = service.verify_task(task_id, Path("workspace"))

        second_evidence = service.verify_task(task_id, Path("workspace"))

        assert second_evidence == first_evidence
        assert verifier.calls == 1

    def test_verify_task_if_pending_ignores_open_work(self) -> None:
        """Do not run verification for a task that is not pending it."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            "Feature",
            "Description",
            FeatureStatus.DRAFT,
        )
        task = repository.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
            TaskStatus.IN_PROGRESS,
        )
        verifier = _FakeVerifier(
            _verification_result(TaskVerificationOutcome.PASSED),
        )
        service = TaskVerificationService(repository, verifier)

        evidence = service.verify_task_if_pending(task.id, Path("workspace"))

        assert evidence is None
        assert verifier.calls == 0

    def test_pending_verification_requires_handoff(self) -> None:
        """Fail clearly when a task has no persisted submission."""
        repository = FakeWorkflowRepository()
        feature = repository.create_feature(
            "Feature",
            "Description",
            FeatureStatus.DRAFT,
        )
        task = repository.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
            TaskStatus.VERIFICATION_PENDING,
        )
        verifier = _FakeVerifier(
            _verification_result(TaskVerificationOutcome.PASSED),
        )
        service = TaskVerificationService(repository, verifier)

        with pytest.raises(TaskSubmissionError):
            service.verify_task(task.id, Path("workspace"))

    def test_workspace_identity_mismatch_fails_closed(self) -> None:
        """Reject verification from a different workspace identity."""
        repository, task_id = _submitted_task(Path("submitted-workspace"))
        verifier = _FakeVerifier(
            _verification_result(TaskVerificationOutcome.PASSED),
        )
        service = TaskVerificationService(repository, verifier)

        with pytest.raises(TaskVerificationWorkspaceError):
            service.verify_task(task_id, Path("verification-workspace"))

        assert verifier.calls == 0

    def test_workspace_identity_accepts_canonical_symlink_root(
        self,
        tmp_path: Path,
    ) -> None:
        """Accept equivalent workspace identities through symlink roots."""
        real_workspace = tmp_path / "workspace"
        real_workspace.mkdir()
        linked_workspace = tmp_path / "linked-workspace"
        linked_workspace.symlink_to(real_workspace, target_is_directory=True)
        repository, task_id = _submitted_task(linked_workspace)
        verifier = _FakeVerifier(
            _verification_result(TaskVerificationOutcome.PASSED),
        )
        service = TaskVerificationService(repository, verifier)

        evidence = service.verify_task(task_id, real_workspace)

        assert evidence.outcome is TaskVerificationOutcome.PASSED
        assert verifier.calls == 1

    def test_missing_workspace_identity_fails_closed(self) -> None:
        """Reject handoffs that have no recoverable workspace identity."""
        repository, task_id = _submitted_task()
        handoff = repository.latest_task_handoff(task_id)
        assert handoff is not None
        repository.handoffs[handoff.id] = replace(
            handoff,
            workspace_identity_hash=None,
        )
        verifier = _FakeVerifier(
            _verification_result(TaskVerificationOutcome.PASSED),
        )
        service = TaskVerificationService(repository, verifier)

        with pytest.raises(TaskVerificationWorkspaceError):
            service.verify_task(task_id, Path("workspace"))

        assert verifier.calls == 0

    def test_legacy_workspace_identity_recovers_from_audit(self) -> None:
        """Use trusted run audit provenance for legacy handoffs."""
        repository, task_id = _submitted_task()
        handoff = repository.latest_task_handoff(task_id)
        assert handoff is not None
        repository.handoffs[handoff.id] = replace(
            handoff,
            workspace_identity_hash=None,
        )
        verifier = _FakeVerifier(
            _verification_result(TaskVerificationOutcome.PASSED),
        )
        audit_reader = FakeAgentAuditReader(
            runs=[
                replace(
                    make_agent_run_record(run_id=handoff.agent_run_id),
                    workspace_identity_hash=workspace_identity_hash(
                        Path("workspace"),
                    ),
                ),
            ],
        )
        service = TaskVerificationService(
            repository,
            verifier,
            audit_reader=audit_reader,
        )

        evidence = service.verify_task(task_id, Path("workspace"))

        assert evidence.outcome is TaskVerificationOutcome.PASSED
        assert verifier.calls == 1

    def test_stale_finalization_is_rejected(self) -> None:
        """Do not persist evidence if status changes during verification."""
        repository, task_id = _submitted_task()
        verifier = _StatusChangingVerifier(
            repository=repository,
            result=_verification_result(TaskVerificationOutcome.PASSED),
        )
        service = TaskVerificationService(repository, verifier)

        with pytest.raises(TaskTransitionError):
            service.verify_task(task_id, Path("workspace"))

        assert verifier.calls == 1
        assert repository.latest_task_verification(task_id) is None


def _submitted_task(
    workspace_root: Path = Path("workspace"),
) -> tuple[FakeWorkflowRepository, int]:
    repository = FakeWorkflowRepository()
    workflow = WorkflowService(repository)
    feature = workflow.create_feature("Feature", "Description")
    task = workflow.create_task(
        feature.id,
        "Task",
        "Description",
        DevelopmentRole.BACKEND_DEVELOPER,
    )
    workflow.update_task_status(task.id, TaskStatus.IN_PROGRESS)
    workflow.submit_task_for_verification(
        _handoff_draft(task.id, workspace_root),
    )
    return repository, task.id


def _handoff_draft(
    task_id: int,
    workspace_root: Path,
) -> TaskHandoffDraft:
    return TaskHandoffDraft(
        task_id=task_id,
        agent_run_id=1,
        submitted_by=DevelopmentRole.BACKEND_DEVELOPER,
        attribution="agent:backend_developer",
        workspace_identity_hash=workspace_identity_hash(workspace_root),
        implementation_summary="Implemented the assigned backend change.",
        changed_paths=("src/app.py",),
        reused_symbols=("ExistingService",),
        new_symbols=("NewHandler",),
        reuse_notes="Existing service handled persistence.",
        checks_attempted=("backend",),
        limitations="none",
        next_action="run deterministic verification",
    )


def _verification_result(
    outcome: TaskVerificationOutcome,
    classification: FailureClassification = (FailureClassification.NONE),
) -> TaskVerificationResult:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return TaskVerificationResult(
        verifier_name="fake-verifier",
        outcome=outcome,
        failure_classification=classification,
        feedback="Fake verification feedback.",
        checks=(
            TaskVerificationCheckResult(
                name="backend",
                started_at=timestamp,
                ended_at=timestamp,
                exit_code=(
                    0 if outcome is TaskVerificationOutcome.PASSED else 1
                ),
                timed_out=False,
                stdout_hash="stdout-hash",
                stdout_excerpt="stdout",
                stderr_hash="stderr-hash",
                stderr_excerpt="stderr",
            ),
        ),
        started_at=timestamp,
        ended_at=timestamp,
    )
