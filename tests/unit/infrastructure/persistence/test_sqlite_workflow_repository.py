"""Tests for SQLite workflow repository."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
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
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


class TestSQLiteWorkflowRepository:
    """SQLite workflow repository behavior tests."""

    def test_creates_parent_directory_and_database(
        self,
        tmp_path: Path,
    ) -> None:
        """Create parent directories and schema automatically."""
        database_path = tmp_path / ".agent_team" / "workflow.db"

        workflow_repository_module.SQLiteWorkflowRepository(database_path)

        assert database_path.exists()

    def test_feature_artifact_and_task_round_trip(
        self,
        tmp_path: Path,
    ) -> None:
        """Persist and reload workflow records."""
        repository = workflow_repository_module.SQLiteWorkflowRepository(
            tmp_path / "workflow.db"
        )

        feature = repository.create_feature(
            title="Build MCP server",
            description="Store development work.",
            status=FeatureStatus.IMPLEMENTATION,
        )
        artifact = repository.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.ARCHITECTURE,
            content="Use SQLite and stdio MCP.",
            created_by="software_architect",
        )
        task = repository.create_task(
            feature_id=feature.id,
            title="Implement SQLite repository",
            description="Create schema and mappings.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )

        assert repository.get_feature(feature.id) == feature
        assert repository.list_features() == [feature]
        assert repository.list_features(FeatureStatus.IMPLEMENTATION) == [
            feature,
        ]
        assert repository.list_artifacts(feature.id) == [artifact]
        assert repository.get_task(task.id) == task
        assert repository.list_tasks(feature.id) == [task]

    def test_update_task_status_persists_status(
        self,
        tmp_path: Path,
    ) -> None:
        """Persist task status updates."""
        repository = workflow_repository_module.SQLiteWorkflowRepository(
            tmp_path / "workflow.db"
        )
        feature = repository.create_feature(
            title="Feature",
            description="Description",
            status=FeatureStatus.DRAFT,
        )
        task = repository.create_task(
            feature_id=feature.id,
            title="Task",
            description="Description",
            assigned_role=DevelopmentRole.QA_ENGINEER,
            status=TaskStatus.PENDING,
        )

        updated_task = repository.update_task_status(
            task_id=task.id,
            status=TaskStatus.IN_PROGRESS,
        )

        assert updated_task is not None
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.updated_at >= task.updated_at
        assert repository.get_task(task.id) == updated_task

    def test_missing_records_return_none_or_empty_lists(
        self,
        tmp_path: Path,
    ) -> None:
        """Return empty values for missing records."""
        repository = workflow_repository_module.SQLiteWorkflowRepository(
            tmp_path / "workflow.db"
        )

        assert repository.get_feature(404) is None
        assert repository.get_task(404) is None
        assert repository.update_task_status(404, TaskStatus.BLOCKED) is None
        assert repository.list_artifacts(404) == []
        assert repository.list_tasks(404) == []

    def test_claim_task_for_work_enforces_active_slot(
        self,
        tmp_path: Path,
    ) -> None:
        """Start one task per feature and role."""
        repository = workflow_repository_module.SQLiteWorkflowRepository(
            tmp_path / "workflow.db"
        )
        feature = repository.create_feature(
            "Feature",
            "Description",
            FeatureStatus.DRAFT,
        )
        first_task = repository.create_task(
            feature.id,
            "First",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
            TaskStatus.PENDING,
        )
        second_task = repository.create_task(
            feature.id,
            "Second",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
            TaskStatus.PENDING,
        )

        claimed = repository.claim_task_for_work(
            first_task.id,
            frozenset({TaskStatus.PENDING}),
            frozenset(
                {
                    TaskStatus.IN_PROGRESS,
                    TaskStatus.VERIFICATION_PENDING,
                },
            ),
        )
        blocked = repository.claim_task_for_work(
            second_task.id,
            frozenset({TaskStatus.PENDING}),
            frozenset(
                {
                    TaskStatus.IN_PROGRESS,
                    TaskStatus.VERIFICATION_PENDING,
                },
            ),
        )

        assert claimed is not None
        assert claimed.status is TaskStatus.IN_PROGRESS
        assert blocked is None

    def test_handoff_and_verification_round_trip(
        self,
        tmp_path: Path,
    ) -> None:
        """Persist structured handoff and deterministic evidence."""
        repository = workflow_repository_module.SQLiteWorkflowRepository(
            tmp_path / "workflow.db"
        )
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

        handoff = repository.submit_task_handoff(
            _handoff_draft(task.id),
            TaskStatus.IN_PROGRESS,
            TaskStatus.VERIFICATION_PENDING,
        )
        assert handoff is not None
        evidence = repository.record_task_verification(
            task_id=task.id,
            submission_id=handoff.id,
            result=_verification_result(),
            next_status=TaskStatus.COMPLETED,
        )
        resumed = repository.record_task_verification(
            task_id=task.id,
            submission_id=handoff.id,
            result=_verification_result(),
            next_status=TaskStatus.COMPLETED,
        )

        assert repository.latest_task_handoff(task.id) == handoff
        assert repository.latest_task_verification(task.id) == evidence
        assert resumed == evidence
        assert evidence.checks[0].name == "backend"
        completed_task = repository.get_task(task.id)
        assert completed_task is not None
        assert completed_task.status is TaskStatus.COMPLETED

    def test_legacy_backend_task_receives_default_contract(
        self,
        tmp_path: Path,
    ) -> None:
        """Migrate old task rows and provide safe default contracts."""
        database_path = tmp_path / "workflow.db"
        _create_legacy_workflow_database(database_path)

        repository = workflow_repository_module.SQLiteWorkflowRepository(
            database_path,
        )
        task = repository.get_task(1)

        assert task is not None
        assert task.verification_contract is not None
        assert task.verification_contract.profile_name == "backend"
        assert task.verification_contract.required_checks == ("backend",)


def _handoff_draft(task_id: int) -> TaskHandoffDraft:
    return TaskHandoffDraft(
        task_id=task_id,
        agent_run_id=1,
        submitted_by=DevelopmentRole.BACKEND_DEVELOPER,
        attribution="agent:backend_developer",
        implementation_summary="Implemented.",
        changed_paths=("src/app.py",),
        reused_symbols=("ExistingService",),
        new_symbols=("NewHandler",),
        reuse_notes="Reused persistence.",
        checks_attempted=("backend",),
        limitations="none",
        next_action="verify",
    )


def _verification_result() -> TaskVerificationResult:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return TaskVerificationResult(
        verifier_name="local_workspace_checks",
        outcome=TaskVerificationOutcome.PASSED,
        failure_classification=FailureClassification.NONE,
        feedback="Passed.",
        checks=(
            TaskVerificationCheckResult(
                name="backend",
                started_at=timestamp,
                ended_at=timestamp,
                exit_code=0,
                timed_out=False,
                stdout_hash="stdout-hash",
                stdout_excerpt="ok",
                stderr_hash="stderr-hash",
                stderr_excerpt="",
            ),
        ),
        started_at=timestamp,
        ended_at=timestamp,
    )


def _create_legacy_workflow_database(database_path: Path) -> None:
    connection = sqlite3.connect(database_path)
    timestamp = "2026-01-01T00:00:00+00:00"
    try:
        connection.executescript(
            """
            CREATE TABLE features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE development_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                assigned_role TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO features (
                id, title, description, status, created_at, updated_at
            )
            VALUES (1, 'Legacy', 'Description', 'draft', ?, ?)
            """,
            (timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO development_tasks (
                id,
                feature_id,
                title,
                description,
                assigned_role,
                status,
                created_at,
                updated_at
            )
            VALUES (
                1,
                1,
                'Legacy task',
                'Description',
                'backend_developer',
                'pending',
                ?,
                ?
            )
            """,
            (timestamp, timestamp),
        )
        connection.commit()
    finally:
        connection.close()
