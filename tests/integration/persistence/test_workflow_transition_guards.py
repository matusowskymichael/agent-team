"""Reject stale workflow operations before they can mutate local state."""

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)

PendingTask = tuple[
    workflow_repository_module.SQLiteWorkflowRepository,
    DevelopmentTask,
]
FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


class TestWorkflowTransitionGuards:
    """Keep failed compare-and-set operations and malformed rows explicit."""

    @pytest.mark.parametrize("empty_from_statuses", [True, False])
    def test_guard_rejects_empty_statuses(
        self,
        persisted_pending_task: PendingTask,
        empty_from_statuses: bool,
    ) -> None:
        """Never claim work under an empty or undefined active-slot policy."""
        repository, task = persisted_pending_task
        from_statuses: frozenset[TaskStatus] = (
            frozenset() if empty_from_statuses else frozenset({task.status})
        )
        active_statuses: frozenset[TaskStatus] = (
            frozenset({TaskStatus.IN_PROGRESS})
            if empty_from_statuses
            else frozenset()
        )

        assert (
            repository.claim_task_for_work(
                task.id,
                from_statuses,
                active_statuses,
            )
            is None
        )
        assert (
            repository.transition_task_status(
                task.id,
                frozenset(),
                TaskStatus.BLOCKED,
            )
            is None
        )
        assert repository.get_task(task.id) == task

    @pytest.mark.parametrize("task_id", [1, 404])
    def test_guard_rejects_incompatible_state(
        self,
        persisted_pending_task: PendingTask,
        persisted_backend_handoff: TaskHandoffDraft,
        task_id: int,
    ) -> None:
        """Avoid mutations for missing tasks or stale expected statuses."""
        repository, task = persisted_pending_task
        operation = Mock(return_value="mutated")

        assert (
            repository.claim_task_for_work(
                task_id,
                frozenset({TaskStatus.BLOCKED}),
                frozenset({TaskStatus.IN_PROGRESS}),
            )
            is None
        )
        assert (
            repository.transition_task_status(
                task_id,
                frozenset({TaskStatus.BLOCKED}),
                TaskStatus.IN_PROGRESS,
            )
            is None
        )
        assert (
            repository.run_task_status_locked(
                task_id,
                TaskStatus.IN_PROGRESS,
                operation,
            )
            is None
        )
        assert (
            repository.submit_task_handoff(
                replace(persisted_backend_handoff, task_id=task_id),
                TaskStatus.IN_PROGRESS,
                TaskStatus.VERIFICATION_PENDING,
            )
            is None
        )

        operation.assert_not_called()
        assert repository.get_task(task.id) == task
        assert repository.latest_task_handoff(task.id) is None

    @pytest.mark.parametrize("submission_id", [1, 404])
    def test_invalid_verification_inputs(
        self,
        persisted_pending_task: PendingTask,
        submission_id: int,
    ) -> None:
        """Persist no verification evidence without the pending submission."""
        repository, task = persisted_pending_task
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        result = TaskVerificationResult(
            verifier_name="trusted",
            outcome=TaskVerificationOutcome.PASSED,
            failure_classification=FailureClassification.NONE,
            feedback="Check passed.",
            checks=(),
            started_at=timestamp,
            ended_at=timestamp,
        )

        assert (
            repository.record_task_verification(
                task_id=task.id,
                submission_id=submission_id,
                result=result,
                next_status=TaskStatus.COMPLETED,
                required_status=TaskStatus.VERIFICATION_PENDING,
                latest_submission_id=submission_id,
            )
            is None
        )

        assert repository.get_task(task.id) == task
        assert repository.latest_task_verification(task.id) is None

    @pytest.mark.parametrize("checks_json", ['{"backend": true}', "[123]"])
    def test_corrupt_contract_json(
        self,
        persisted_pending_task: PendingTask,
        sqlite_connection: Callable[[Path], sqlite3.Connection],
        checks_json: str,
    ) -> None:
        """Reject malformed persisted trusted-check lists on loading."""
        repository, task = persisted_pending_task
        connection = sqlite_connection(repository.database_path)
        connection.execute(
            "UPDATE development_tasks SET required_checks_json = ? "
            "WHERE id = ?",
            (checks_json, task.id),
        )
        connection.commit()

        with pytest.raises(RuntimeError, match="invalid workflow JSON"):
            repository.get_task(task.id)
