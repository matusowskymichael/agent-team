"""Tests for workflow application service."""

import pytest

from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.development_task_not_found_error import (
    DevelopmentTaskNotFoundError,
)
from agent_team.domain.workflow.feature_not_found_error import (
    FeatureNotFoundError,
)
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.workflow_validation_error import (
    WorkflowValidationError,
)
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestWorkflowService:
    """Workflow service behavior tests."""

    def test_workflow_operations_use_repository(self) -> None:
        """Create and read workflow records through the repository port."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)

        feature = service.create_feature(
            title="  Build MCP server  ",
            description="  Store development work.  ",
            status="analysis",
        )
        artifact = service.add_artifact(
            feature_id=feature.id,
            kind="requirements",
            content="Persist features and tasks.",
            created_by="business_analyst",
        )
        task = service.create_task(
            feature_id=feature.id,
            title="Implement repository",
            description="Create SQLite adapter.",
            assigned_role="backend_developer",
        )
        updated_task = service.update_task_status(
            task_id=task.id,
            status="completed",
        )

        assert feature.title == "Build MCP server"
        assert feature.description == "Store development work."
        assert feature.status == FeatureStatus.ANALYSIS
        assert service.get_feature(feature.id) == feature
        assert service.list_features(FeatureStatus.ANALYSIS) == [feature]
        assert artifact.kind == ArtifactKind.REQUIREMENTS
        assert service.list_artifacts(feature.id) == [artifact]
        assert task.assigned_role == DevelopmentRole.BACKEND_DEVELOPER
        assert updated_task.status == TaskStatus.COMPLETED
        assert service.list_tasks(feature.id) == [updated_task]
        overview = service.get_feature_overview(feature.id)
        assert overview.feature == feature
        assert overview.artifacts == (artifact,)
        assert overview.tasks == (updated_task,)

    def test_overview_includes_artifacts_and_empty_tasks(self) -> None:
        """Return artifacts and an explicit empty task collection."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature(
            title="User Authentication",
            description="Secure login and logout.",
        )
        artifact = service.add_artifact(
            feature_id=feature.id,
            kind=ArtifactKind.REQUIREMENTS,
            content="Users need secure login.",
            created_by="agent:business_analyst",
        )

        overview = service.get_feature_overview(feature.id)

        assert overview.feature == feature
        assert overview.artifacts == (artifact,)
        assert overview.tasks == ()

    @pytest.mark.parametrize(
        ("title", "description", "field_name"),
        [
            ("", "Valid description.", "title"),
            ("   ", "Valid description.", "title"),
            ("Valid title.", "", "description"),
            ("Valid title.", "   ", "description"),
        ],
    )
    def test_create_feature_rejects_blank_required_text(
        self,
        title: str,
        description: str,
        field_name: str,
    ) -> None:
        """Reject blank feature text."""
        service = WorkflowService(repository=FakeWorkflowRepository())

        with pytest.raises(WorkflowValidationError) as error:
            service.create_feature(title=title, description=description)

        assert field_name in str(error.value)

    def test_create_feature_rejects_invalid_status(self) -> None:
        """Reject invalid feature status values."""
        service = WorkflowService(repository=FakeWorkflowRepository())

        with pytest.raises(WorkflowValidationError) as error:
            service.create_feature(
                title="Feature",
                description="Description",
                status="invalid",
            )

        assert "status" in str(error.value)
        assert "draft" in str(error.value)

    def test_add_artifact_requires_existing_feature(self) -> None:
        """Reject artifacts for missing features."""
        service = WorkflowService(repository=FakeWorkflowRepository())

        with pytest.raises(FeatureNotFoundError):
            service.add_artifact(
                feature_id=404,
                kind=ArtifactKind.REQUIREMENTS,
                content="Requirements",
                created_by="business_analyst",
            )

    def test_create_task_rejects_invalid_role(self) -> None:
        """Reject invalid development roles."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature(
            title="Feature",
            description="Description",
        )

        with pytest.raises(WorkflowValidationError) as error:
            service.create_task(
                feature_id=feature.id,
                title="Task",
                description="Description",
                assigned_role="invalid",
            )

        assert "assigned_role" in str(error.value)
        assert "backend_developer" in str(error.value)

    def test_list_artifacts_requires_existing_feature(self) -> None:
        """Reject listing artifacts for missing features."""
        service = WorkflowService(repository=FakeWorkflowRepository())

        with pytest.raises(FeatureNotFoundError):
            service.list_artifacts(feature_id=404)

    def test_update_task_status_requires_existing_task(self) -> None:
        """Reject status updates for missing tasks."""
        service = WorkflowService(repository=FakeWorkflowRepository())

        with pytest.raises(DevelopmentTaskNotFoundError):
            service.update_task_status(task_id=404, status=TaskStatus.BLOCKED)
