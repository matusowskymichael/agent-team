"""Workflow repository port."""

from typing import Protocol

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus


class WorkflowRepository(Protocol):
    """Persistence port for development workflow data."""

    def create_feature(
        self,
        title: str,
        description: str,
        status: FeatureStatus,
    ) -> Feature:
        """Create and persist a feature."""
        ...

    def get_feature(self, feature_id: int) -> Feature | None:
        """Return a feature by ID, if it exists."""
        ...

    def list_features(
        self,
        status: FeatureStatus | None = None,
    ) -> list[Feature]:
        """Return persisted features, optionally filtered by status."""
        ...

    def add_artifact(
        self,
        feature_id: int,
        kind: ArtifactKind,
        content: str,
        created_by: str,
    ) -> Artifact:
        """Create and persist a feature artifact."""
        ...

    def list_artifacts(self, feature_id: int) -> list[Artifact]:
        """Return artifacts attached to a feature."""
        ...

    def create_task(
        self,
        feature_id: int,
        title: str,
        description: str,
        assigned_role: DevelopmentRole,
        status: TaskStatus,
    ) -> DevelopmentTask:
        """Create and persist a development task."""
        ...

    def get_task(self, task_id: int) -> DevelopmentTask | None:
        """Return a development task by ID, if it exists."""
        ...

    def list_tasks(self, feature_id: int) -> list[DevelopmentTask]:
        """Return development tasks attached to a feature."""
        ...

    def update_task_status(
        self,
        task_id: int,
        status: TaskStatus,
    ) -> DevelopmentTask | None:
        """Update a task status and return the updated task, if it exists."""
        ...
