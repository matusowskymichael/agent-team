"""Application service for development workflow operations."""

from dataclasses import dataclass
from enum import StrEnum

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.development_task_not_found_error import (
    DevelopmentTaskNotFoundError,
)
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_not_found_error import (
    FeatureNotFoundError,
)
from agent_team.domain.workflow.feature_overview import FeatureOverview
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.workflow_repository import WorkflowRepository
from agent_team.domain.workflow.workflow_validation_error import (
    WorkflowValidationError,
)


@dataclass(frozen=True, slots=True)
class WorkflowService:
    """Use cases for managing development workflow records."""

    repository: WorkflowRepository

    def create_feature(
        self,
        title: str,
        description: str,
        status: FeatureStatus | str = FeatureStatus.DRAFT,
    ) -> Feature:
        """Create a feature after validating workflow input."""
        clean_title = _require_text(title, "title")
        clean_description = _require_text(description, "description")
        feature_status = _parse_enum(status, FeatureStatus, "status")
        return self.repository.create_feature(
            title=clean_title,
            description=clean_description,
            status=feature_status,
        )

    def get_feature(self, feature_id: int) -> Feature:
        """Return an existing feature."""
        feature = self.repository.get_feature(feature_id)
        if feature is None:
            raise FeatureNotFoundError(f"Feature {feature_id} was not found.")
        return feature

    def get_feature_overview(self, feature_id: int) -> FeatureOverview:
        """Return a feature with all attached artifacts and tasks."""
        feature = self._require_feature(feature_id)
        artifacts = self.repository.list_artifacts(feature_id)
        tasks = self.repository.list_tasks(feature_id)
        return FeatureOverview(
            feature=feature,
            artifacts=tuple(artifacts),
            tasks=tuple(tasks),
        )

    def list_features(
        self,
        status: FeatureStatus | str | None = None,
    ) -> list[Feature]:
        """Return features, optionally filtered by status."""
        if status is None:
            feature_status = None
        else:
            feature_status = _parse_enum(status, FeatureStatus, "status")
        return self.repository.list_features(feature_status)

    def add_artifact(
        self,
        feature_id: int,
        kind: ArtifactKind | str,
        content: str,
        created_by: str,
    ) -> Artifact:
        """Add an artifact to an existing feature."""
        self._require_feature(feature_id)
        artifact_kind = _parse_enum(kind, ArtifactKind, "kind")
        clean_content = _require_text(content, "content")
        clean_created_by = _require_text(created_by, "created_by")
        return self.repository.add_artifact(
            feature_id=feature_id,
            kind=artifact_kind,
            content=clean_content,
            created_by=clean_created_by,
        )

    def list_artifacts(self, feature_id: int) -> list[Artifact]:
        """Return artifacts attached to an existing feature."""
        self._require_feature(feature_id)
        return self.repository.list_artifacts(feature_id)

    def create_task(
        self,
        feature_id: int,
        title: str,
        description: str,
        assigned_role: DevelopmentRole | str,
        status: TaskStatus | str = TaskStatus.PENDING,
    ) -> DevelopmentTask:
        """Create a task for an existing feature."""
        self._require_feature(feature_id)
        clean_title = _require_text(title, "title")
        clean_description = _require_text(description, "description")
        role = _parse_enum(assigned_role, DevelopmentRole, "assigned_role")
        task_status = _parse_enum(status, TaskStatus, "status")
        return self.repository.create_task(
            feature_id=feature_id,
            title=clean_title,
            description=clean_description,
            assigned_role=role,
            status=task_status,
        )

    def list_tasks(self, feature_id: int) -> list[DevelopmentTask]:
        """Return tasks attached to an existing feature."""
        self._require_feature(feature_id)
        return self.repository.list_tasks(feature_id)

    def update_task_status(
        self,
        task_id: int,
        status: TaskStatus | str,
    ) -> DevelopmentTask:
        """Update an existing task status."""
        self._require_task(task_id)
        task_status = _parse_enum(status, TaskStatus, "status")
        updated_task = self.repository.update_task_status(task_id, task_status)
        if updated_task is None:
            raise DevelopmentTaskNotFoundError(
                f"Development task {task_id} was not found.",
            )
        return updated_task

    def _require_feature(self, feature_id: int) -> Feature:
        feature = self.repository.get_feature(feature_id)
        if feature is None:
            raise FeatureNotFoundError(f"Feature {feature_id} was not found.")
        return feature

    def _require_task(self, task_id: int) -> DevelopmentTask:
        task = self.repository.get_task(task_id)
        if task is None:
            raise DevelopmentTaskNotFoundError(
                f"Development task {task_id} was not found.",
            )
        return task


def _require_text(value: str, field_name: str) -> str:
    clean_value = value.strip()
    if not clean_value:
        raise WorkflowValidationError(f"{field_name} must not be blank.")
    return clean_value


def _parse_enum[EnumValue: StrEnum](
    value: EnumValue | str,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as error:
        valid_values = ", ".join(item.value for item in enum_type)
        message = f"{field_name} must be one of: {valid_values}."
        raise WorkflowValidationError(message) from error
