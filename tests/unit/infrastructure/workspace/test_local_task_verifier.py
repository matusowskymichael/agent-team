"""Tests for the local deterministic task verifier."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_contract import (
    TaskVerificationContract,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_profiles import (
    BACKEND_REQUIRED_CHECKS,
)
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_executor import WorkspaceExecutor
from agent_team.infrastructure.workspace.local_task_verifier import (
    LocalTaskVerifier,
)

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


class _FakeExecutor:
    def __init__(
        self,
        result: CheckRunResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or CheckRunResult(
            name="backend",
            exit_code=0,
            stdout_excerpt="ok",
            stderr_excerpt="",
            timed_out=False,
        )
        self.error = error
        self.received_names: list[str] = []

    def run_check(self, name: str) -> CheckRunResult:
        """Return or raise the configured check result."""
        self.received_names.append(name)
        if self.error is not None:
            raise self.error
        return replace(self.result, name=name)


class TestLocalTaskVerifier:
    """LocalTaskVerifier behavior tests."""

    def test_passes_when_required_checks_pass(self) -> None:
        """Return a passed verification result."""
        executor = _FakeExecutor()
        verifier = LocalTaskVerifier(executor_factory=_factory(executor))

        result = verifier.verify(_task(), _handoff(), Path("workspace"))

        assert result.outcome is TaskVerificationOutcome.PASSED
        assert result.failure_classification is (FailureClassification.NONE)
        assert result.checks[0].stdout_hash
        assert executor.received_names == list(BACKEND_REQUIRED_CHECKS)

    def test_failed_check_returns_check_failed(self) -> None:
        """Return failed verification when a check exits non-zero."""
        executor = _FakeExecutor(
            CheckRunResult(
                name="backend",
                exit_code=1,
                stdout_excerpt="",
                stderr_excerpt="failed",
                timed_out=False,
            ),
        )
        verifier = LocalTaskVerifier(executor_factory=_factory(executor))

        result = verifier.verify(_task(), _handoff(), Path("workspace"))

        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.failure_classification is (
            FailureClassification.CHECK_FAILED
        )
        assert "failed" in result.feedback

    def test_timeout_returns_timeout_failure(self) -> None:
        """Classify timed-out checks separately."""
        executor = _FakeExecutor(
            CheckRunResult(
                name="backend",
                exit_code=124,
                stdout_excerpt="",
                stderr_excerpt="timeout",
                timed_out=True,
            ),
        )
        verifier = LocalTaskVerifier(executor_factory=_factory(executor))

        result = verifier.verify(_task(), _handoff(), Path("workspace"))

        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.failure_classification is (FailureClassification.TIMEOUT)

    def test_missing_required_checks_blocks_verification(self) -> None:
        """Block verification for missing trusted checks."""
        task = _task(TaskVerificationContract("backend", ()))
        verifier = LocalTaskVerifier(
            executor_factory=_factory(_FakeExecutor()),
        )

        result = verifier.verify(task, _handoff(), Path("workspace"))

        assert result.outcome is TaskVerificationOutcome.BLOCKED
        assert result.failure_classification is (
            FailureClassification.CONFIGURATION_ERROR
        )

    def test_denied_check_configuration_blocks_verification(self) -> None:
        """Treat workspace access denials as configuration blockers."""
        executor = _FakeExecutor(
            error=WorkspaceAccessDeniedError("Workspace check is denied."),
        )
        verifier = LocalTaskVerifier(executor_factory=_factory(executor))

        result = verifier.verify(_task(), _handoff(), Path("workspace"))

        assert result.outcome is TaskVerificationOutcome.BLOCKED
        assert result.failure_classification is (
            FailureClassification.CONFIGURATION_ERROR
        )

    def test_executor_infrastructure_error_blocks_verification(self) -> None:
        """Treat local executor failures as infrastructure blockers."""
        verifier = LocalTaskVerifier(
            executor_factory=_raising_factory,
        )

        result = verifier.verify(_task(), _handoff(), Path("workspace"))

        assert result.outcome is TaskVerificationOutcome.BLOCKED
        assert result.failure_classification is (
            FailureClassification.INFRASTRUCTURE_ERROR
        )


def _task(
    contract: TaskVerificationContract | None = None,
) -> DevelopmentTask:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return DevelopmentTask(
        id=1,
        feature_id=1,
        title="Task",
        description="Description",
        assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
        status=TaskStatus.VERIFICATION_PENDING,
        created_at=timestamp,
        updated_at=timestamp,
        verification_contract=contract
        or TaskVerificationContract("backend", BACKEND_REQUIRED_CHECKS),
    )


def _handoff() -> TaskHandoff:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return TaskHandoff(
        id=1,
        task_id=1,
        agent_run_id=1,
        submitted_by=DevelopmentRole.BACKEND_DEVELOPER,
        attribution="agent:backend_developer",
        workspace_identity_hash="workspace-hash",
        implementation_summary="Implemented.",
        changed_paths=("src/app.py",),
        reused_symbols=("ExistingService",),
        new_symbols=("NewHandler",),
        reuse_notes="Reused persistence.",
        checks_attempted=("backend",),
        limitations="none",
        next_action="verify",
        created_at=timestamp,
    )


def _raise_os_error() -> _FakeExecutor:
    raise OSError("workspace unavailable")


def _factory(executor: _FakeExecutor) -> Callable[[Path], WorkspaceExecutor]:
    def create_executor(_root: Path) -> WorkspaceExecutor:
        return cast("WorkspaceExecutor", executor)

    return create_executor


def _raising_factory(_root: Path) -> WorkspaceExecutor:
    return cast("WorkspaceExecutor", _raise_os_error())
