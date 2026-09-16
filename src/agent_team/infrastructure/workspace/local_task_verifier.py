"""Local deterministic task verifier."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_team.application.audit.audit_sanitizer import (
    hash_text,
    sanitize_error,
)
from agent_team.application.workflow.task_verifier import TaskVerifier
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_check_result import (
    TaskVerificationCheckResult,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_executor import WorkspaceExecutor

VERIFIER_NAME = "local_workspace_checks"
FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


@dataclass(frozen=True, slots=True)
class LocalTaskVerifier(TaskVerifier):
    """Run task verification through trusted local workspace checks."""

    executor_factory: Callable[[Path], WorkspaceExecutor]
    verifier_name: str = VERIFIER_NAME

    def verify(
        self,
        task: DevelopmentTask,
        handoff: TaskHandoff,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        """Run the task's trusted verification checks."""
        del handoff
        started_at = _utc_now()
        contract = task.verification_contract
        if contract is None or not contract.required_checks:
            return _blocked(
                self.verifier_name,
                started_at,
                (),
                "Task verification configuration is missing required checks.",
                FailureClassification.CONFIGURATION_ERROR,
            )
        try:
            executor = self.executor_factory(workspace_root)
        except (OSError, RuntimeError) as error:
            return _blocked_from_error(
                self.verifier_name,
                started_at,
                (),
                error,
            )

        checks, failure = _run_checks(
            self.verifier_name,
            started_at,
            executor,
            contract.required_checks,
        )
        if failure is not None:
            return failure

        return TaskVerificationResult(
            verifier_name=self.verifier_name,
            outcome=TaskVerificationOutcome.PASSED,
            failure_classification=FailureClassification.NONE,
            feedback="All required verification checks passed.",
            checks=tuple(checks),
            started_at=started_at,
            ended_at=_utc_now(),
        )


def _blocked_from_error(
    verifier_name: str,
    started_at: datetime,
    checks: tuple[TaskVerificationCheckResult, ...],
    error: Exception,
) -> TaskVerificationResult:
    return _blocked(
        verifier_name,
        started_at,
        checks,
        _error_feedback(error),
        FailureClassification.INFRASTRUCTURE_ERROR,
    )


def _run_checks(
    verifier_name: str,
    started_at: datetime,
    executor: WorkspaceExecutor,
    check_names: tuple[str, ...],
) -> tuple[
    tuple[TaskVerificationCheckResult, ...],
    TaskVerificationResult | None,
]:
    checks: list[TaskVerificationCheckResult] = []
    for check_name in check_names:
        check_started_at = _utc_now()
        try:
            check = executor.run_check(check_name)
        except WorkspaceAccessDeniedError as error:
            return tuple(checks), _blocked(
                verifier_name,
                started_at,
                tuple(checks),
                _error_feedback(error),
                FailureClassification.CONFIGURATION_ERROR,
            )
        except (OSError, RuntimeError, UnicodeDecodeError) as error:
            return tuple(checks), _blocked_from_error(
                verifier_name,
                started_at,
                tuple(checks),
                error,
            )
        check_result = _check_result(check, check_started_at, _utc_now())
        checks.append(check_result)
        failure = _check_failure(
            verifier_name,
            started_at,
            check_name,
            tuple(checks),
            check_result,
        )
        if failure is not None:
            return tuple(checks), failure
    return tuple(checks), None


def _check_failure(
    verifier_name: str,
    started_at: datetime,
    check_name: str,
    checks: tuple[TaskVerificationCheckResult, ...],
    check: TaskVerificationCheckResult,
) -> TaskVerificationResult | None:
    if check.timed_out:
        return _failed(
            verifier_name,
            started_at,
            checks,
            FailureClassification.TIMEOUT,
            f"Verification check {check_name} timed out.",
        )
    if check.exit_code != 0:
        return _failed(
            verifier_name,
            started_at,
            checks,
            FailureClassification.CHECK_FAILED,
            f"Verification check {check_name} failed.",
        )
    return None


def _blocked(
    verifier_name: str,
    started_at: datetime,
    checks: tuple[TaskVerificationCheckResult, ...],
    feedback: str,
    classification: FailureClassification,
) -> TaskVerificationResult:
    return TaskVerificationResult(
        verifier_name=verifier_name,
        outcome=TaskVerificationOutcome.BLOCKED,
        failure_classification=classification,
        feedback=feedback,
        checks=checks,
        started_at=started_at,
        ended_at=_utc_now(),
    )


def _failed(
    verifier_name: str,
    started_at: datetime,
    checks: tuple[TaskVerificationCheckResult, ...],
    classification: FailureClassification,
    feedback: str,
) -> TaskVerificationResult:
    return TaskVerificationResult(
        verifier_name=verifier_name,
        outcome=TaskVerificationOutcome.FAILED,
        failure_classification=classification,
        feedback=f"{feedback} Fix the issue and rerun the same trusted check.",
        checks=checks,
        started_at=started_at,
        ended_at=_utc_now(),
    )


def _check_result(
    check: CheckRunResult,
    started_at: datetime,
    ended_at: datetime,
) -> TaskVerificationCheckResult:
    return TaskVerificationCheckResult(
        name=check.name,
        started_at=started_at,
        ended_at=ended_at,
        exit_code=check.exit_code,
        timed_out=check.timed_out,
        stdout_hash=hash_text(check.stdout_excerpt),
        stdout_excerpt=check.stdout_excerpt,
        stderr_hash=hash_text(check.stderr_excerpt),
        stderr_excerpt=check.stderr_excerpt,
    )


def _error_feedback(error: Exception) -> str:
    error_type, message = sanitize_error(error)
    return f"Verification could not run: {error_type}: {message}."


def _utc_now() -> datetime:
    return datetime.now(UTC)
