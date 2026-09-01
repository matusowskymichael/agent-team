"""Fake workflow repository for unit tests."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus


def _feature_store() -> dict[int, Feature]:
    return {}


def _artifact_store() -> dict[int, Artifact]:
    return {}


def _task_store() -> dict[int, DevelopmentTask]:
    return {}


@dataclass(slots=True)
class FakeWorkflowRepository:
    """In-memory workflow repository fake."""

    features: dict[int, Feature] = field(default_factory=_feature_store)
    artifacts: dict[int, Artifact] = field(default_factory=_artifact_store)
    tasks: dict[int, DevelopmentTask] = field(default_factory=_task_store)
    next_feature_id: int = 1
    next_artifact_id: int = 1
    next_task_id: int = 1

    def create_feature(
        self,
        title: str,
        description: str,
        status: FeatureStatus,
    ) -> Feature:
        """Create and store a fake feature."""
        timestamp = _timestamp()
        feature = Feature(
            id=self.next_feature_id,
            title=title,
            description=description,
            status=status,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.features[feature.id] = feature
        self.next_feature_id += 1
        return feature

    def get_feature(self, feature_id: int) -> Feature | None:
        """Return a fake feature by ID."""
        return self.features.get(feature_id)

    def list_features(
        self,
        status: FeatureStatus | None = None,
    ) -> list[Feature]:
        """Return fake features, optionally filtered by status."""
        features = sorted(
            self.features.values(),
            key=lambda feature: feature.id,
        )
        if status is None:
            return features
        return [feature for feature in features if feature.status == status]

    def add_artifact(
        self,
        feature_id: int,
        kind: ArtifactKind,
        content: str,
        created_by: str,
    ) -> Artifact:
        """Create and store a fake artifact."""
        artifact = Artifact(
            id=self.next_artifact_id,
            feature_id=feature_id,
            kind=kind,
            content=content,
            created_by=created_by,
            created_at=_timestamp(),
        )
        self.artifacts[artifact.id] = artifact
        self.next_artifact_id += 1
        return artifact

    def list_artifacts(self, feature_id: int) -> list[Artifact]:
        """Return fake artifacts for a feature."""
        artifacts = sorted(
            self.artifacts.values(),
            key=lambda artifact: artifact.id,
        )
        return [
            artifact
            for artifact in artifacts
            if artifact.feature_id == feature_id
        ]

    def create_task(
        self,
        feature_id: int,
        title: str,
        description: str,
        assigned_role: DevelopmentRole,
        status: TaskStatus,
    ) -> DevelopmentTask:
        """Create and store a fake development task."""
        timestamp = _timestamp()
        task = DevelopmentTask(
            id=self.next_task_id,
            feature_id=feature_id,
            title=title,
            description=description,
            assigned_role=assigned_role,
            status=status,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.tasks[task.id] = task
        self.next_task_id += 1
        return task

    def get_task(self, task_id: int) -> DevelopmentTask | None:
        """Return a fake task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(self, feature_id: int) -> list[DevelopmentTask]:
        """Return fake tasks for a feature."""
        tasks = sorted(self.tasks.values(), key=lambda task: task.id)
        return [task for task in tasks if task.feature_id == feature_id]

    def update_task_status(
        self,
        task_id: int,
        status: TaskStatus,
    ) -> DevelopmentTask | None:
        """Update and return a fake task status."""
        task = self.tasks.get(task_id)
        if task is None:
            return None
        updated_task = replace(
            task,
            status=status,
            updated_at=_timestamp(),
        )
        self.tasks[task_id] = updated_task
        return updated_task


def _timestamp() -> datetime:
    return datetime.now(UTC)
