"""Fake workflow repository for unit tests."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_check import (
    TaskVerificationCheck,
)
from agent_team.domain.workflow.task_verification_evidence import (
    TaskVerificationEvidence,
)
from agent_team.domain.workflow.task_verification_profiles import (
    default_verification_contract,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)


def _feature_store() -> dict[int, Feature]:
    return {}


def _artifact_store() -> dict[int, Artifact]:
    return {}


def _task_store() -> dict[int, DevelopmentTask]:
    return {}


def _handoff_store() -> dict[int, TaskHandoff]:
    return {}


def _verification_store() -> dict[int, TaskVerificationEvidence]:
    return {}


@dataclass(slots=True)
class FakeWorkflowRepository:
    """In-memory workflow repository fake."""

    features: dict[int, Feature] = field(default_factory=_feature_store)
    artifacts: dict[int, Artifact] = field(default_factory=_artifact_store)
    tasks: dict[int, DevelopmentTask] = field(default_factory=_task_store)
    handoffs: dict[int, TaskHandoff] = field(default_factory=_handoff_store)
    verifications: dict[int, TaskVerificationEvidence] = field(
        default_factory=_verification_store,
    )
    next_feature_id: int = 1
    next_artifact_id: int = 1
    next_task_id: int = 1
    next_handoff_id: int = 1
    next_verification_id: int = 1
    next_verification_check_id: int = 1

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
            verification_contract=default_verification_contract(
                assigned_role,
            ),
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

    def claim_task_for_work(
        self,
        task_id: int,
        from_statuses: frozenset[TaskStatus],
        active_statuses: frozenset[TaskStatus],
    ) -> DevelopmentTask | None:
        """Start one task if no same-role task is already active."""
        task = self.tasks.get(task_id)
        if task is None or task.status not in from_statuses:
            return None
        conflict = any(
            candidate.id != task_id
            and candidate.feature_id == task.feature_id
            and candidate.assigned_role is task.assigned_role
            and candidate.status in active_statuses
            for candidate in self.tasks.values()
        )
        if conflict:
            return None
        return self.update_task_status(task_id, TaskStatus.IN_PROGRESS)

    def transition_task_status(
        self,
        task_id: int,
        from_statuses: frozenset[TaskStatus],
        to_status: TaskStatus,
    ) -> DevelopmentTask | None:
        """Compare-and-set a task status transition."""
        task = self.tasks.get(task_id)
        if task is None or task.status not in from_statuses:
            return None
        return self.update_task_status(task_id, to_status)

    def submit_task_handoff(
        self,
        draft: TaskHandoffDraft,
        from_status: TaskStatus,
        to_status: TaskStatus,
    ) -> TaskHandoff | None:
        """Persist a handoff and move the task to verification."""
        task = self.tasks.get(draft.task_id)
        if task is None or task.status is not from_status:
            return None
        timestamp = _timestamp()
        handoff = TaskHandoff(
            id=self.next_handoff_id,
            task_id=draft.task_id,
            agent_run_id=draft.agent_run_id,
            submitted_by=draft.submitted_by,
            attribution=draft.attribution,
            implementation_summary=draft.implementation_summary,
            changed_paths=draft.changed_paths,
            reused_symbols=draft.reused_symbols,
            new_symbols=draft.new_symbols,
            reuse_notes=draft.reuse_notes,
            checks_attempted=draft.checks_attempted,
            limitations=draft.limitations,
            next_action=draft.next_action,
            created_at=timestamp,
        )
        self.handoffs[handoff.id] = handoff
        self.next_handoff_id += 1
        self.update_task_status(draft.task_id, to_status)
        return handoff

    def latest_task_handoff(self, task_id: int) -> TaskHandoff | None:
        """Return the newest persisted handoff for a task, if present."""
        handoffs = [
            handoff
            for handoff in self.handoffs.values()
            if handoff.task_id == task_id
        ]
        if not handoffs:
            return None
        return max(handoffs, key=lambda handoff: handoff.id)

    def record_task_verification(
        self,
        task_id: int,
        submission_id: int,
        result: TaskVerificationResult,
        next_status: TaskStatus,
    ) -> TaskVerificationEvidence:
        """Persist verification evidence and apply the resulting status."""
        verification_id = self.next_verification_id
        checks: list[TaskVerificationCheck] = []
        for check_result in result.checks:
            check = TaskVerificationCheck(
                id=self.next_verification_check_id,
                verification_id=verification_id,
                name=check_result.name,
                started_at=check_result.started_at,
                ended_at=check_result.ended_at,
                exit_code=check_result.exit_code,
                timed_out=check_result.timed_out,
                stdout_hash=check_result.stdout_hash,
                stdout_excerpt=check_result.stdout_excerpt,
                stderr_hash=check_result.stderr_hash,
                stderr_excerpt=check_result.stderr_excerpt,
            )
            checks.append(check)
            self.next_verification_check_id += 1
        evidence = TaskVerificationEvidence(
            id=verification_id,
            task_id=task_id,
            submission_id=submission_id,
            verifier_name=result.verifier_name,
            outcome=result.outcome,
            failure_classification=result.failure_classification,
            feedback=result.feedback,
            checks=tuple(checks),
            started_at=result.started_at,
            ended_at=result.ended_at,
        )
        self.verifications[evidence.id] = evidence
        self.next_verification_id += 1
        self.update_task_status(task_id, next_status)
        return evidence

    def latest_task_verification(
        self,
        task_id: int,
    ) -> TaskVerificationEvidence | None:
        """Return the newest verification evidence for a task."""
        verifications = [
            verification
            for verification in self.verifications.values()
            if verification.task_id == task_id
        ]
        if not verifications:
            return None
        return max(verifications, key=lambda verification: verification.id)


def _timestamp() -> datetime:
    return datetime.now(UTC)
