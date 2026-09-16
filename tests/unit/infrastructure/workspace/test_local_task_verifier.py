"""Tests for the local deterministic task verifier."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

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
    FRONTEND_REQUIRED_CHECKS,
    default_verification_contract,
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
        fail_on_check: str | None = None,
    ) -> None:
        self.result = result or CheckRunResult(
            name="backend",
            exit_code=0,
            stdout_excerpt="ok",
            stderr_excerpt="",
            timed_out=False,
        )
        self.error = error
        self.fail_on_check = fail_on_check
        self.received_names: list[str] = []

    def run_check(self, name: str) -> CheckRunResult:
        """Return or raise the configured check result."""
        self.received_names.append(name)
        if self.error is not None:
            raise self.error
        return replace(
            self.result,
            name=name,
            exit_code=(
                1 if name == self.fail_on_check else self.result.exit_code
            ),
        )


class TestLocalTaskVerifier:
    """LocalTaskVerifier behavior tests."""

    @pytest.mark.parametrize(
        ("role", "required_checks"),
        (
            (DevelopmentRole.BACKEND_DEVELOPER, BACKEND_REQUIRED_CHECKS),
            (DevelopmentRole.FRONTEND_DEVELOPER, FRONTEND_REQUIRED_CHECKS),
        ),
    )
    def test_passes_when_required_checks_pass(
        self,
        role: DevelopmentRole,
        required_checks: tuple[str, ...],
    ) -> None:
        """Return a passed verification result."""
        executor = _FakeExecutor()
        verifier = LocalTaskVerifier(executor_factory=_factory(executor))
        task = replace(
            _task(),
            assigned_role=role,
            verification_contract=default_verification_contract(role),
        )

        result = verifier.verify(task, _handoff(), Path("workspace"))

        assert result.outcome is TaskVerificationOutcome.PASSED
        assert result.failure_classification is (FailureClassification.NONE)
        assert result.checks[0].stdout_hash
        assert executor.received_names == list(required_checks)
        assert tuple(check.name for check in result.checks) == required_checks

    @pytest.mark.parametrize(
        ("role", "check_name"),
        [
            (role, check_name)
            for role, required_checks in (
                (DevelopmentRole.BACKEND_DEVELOPER, BACKEND_REQUIRED_CHECKS),
                (DevelopmentRole.FRONTEND_DEVELOPER, FRONTEND_REQUIRED_CHECKS),
            )
            for check_name in required_checks
        ],
    )
    def test_each_required_check_can_fail(
        self,
        role: DevelopmentRole,
        check_name: str,
    ) -> None:
        """Prevent completion for a failure in every role's required check."""
        executor = _FakeExecutor(fail_on_check=check_name)
        verifier = LocalTaskVerifier(executor_factory=_factory(executor))
        contract = default_verification_contract(role)
        assert contract is not None
        task = replace(
            _task(),
            assigned_role=role,
            verification_contract=contract,
        )

        result = verifier.verify(task, _handoff(), Path("workspace"))

        expected_checks = contract.required_checks[
            : contract.required_checks.index(check_name) + 1
        ]
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.failure_classification is (
            FailureClassification.CHECK_FAILED
        )
        assert check_name in result.feedback
        assert tuple(check.name for check in result.checks) == expected_checks
        assert result.checks[-1].exit_code == 1
        assert all(check.exit_code == 0 for check in result.checks[:-1])
        assert executor.received_names == list(expected_checks)

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

    @pytest.mark.parametrize(
        "error",
        (
            OSError("token=private-value"),
            RuntimeError("token=private-value"),
            UnicodeDecodeError(
                "utf-8",
                b"\xff",
                0,
                1,
                "token=private-value",
            ),
        ),
        ids=("os-error", "runtime-error", "decode-error"),
    )
    def test_check_infrastructure_error_blocks_verification(
        self,
        error: Exception,
    ) -> None:
        """Return sanitized blocked evidence when check execution fails."""
        executor = _FakeExecutor(error=error)
        verifier = LocalTaskVerifier(executor_factory=_factory(executor))

        result = verifier.verify(_task(), _handoff(), Path("workspace"))

        assert result.outcome is TaskVerificationOutcome.BLOCKED
        assert result.failure_classification is (
            FailureClassification.INFRASTRUCTURE_ERROR
        )
        assert result.checks == ()
        assert type(error).__name__ in result.feedback
        assert "[REDACTED]" in result.feedback
        assert "private-value" not in result.feedback
        assert executor.received_names == [BACKEND_REQUIRED_CHECKS[0]]


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
