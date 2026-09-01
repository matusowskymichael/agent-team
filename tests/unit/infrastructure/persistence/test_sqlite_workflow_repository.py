"""Tests for SQLite workflow repository."""

from pathlib import Path

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
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
