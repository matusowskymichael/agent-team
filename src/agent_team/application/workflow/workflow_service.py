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
from agent_team.domain.workflow.task_active_slot_conflict_error import (
    TaskActiveSlotConflictError,
)
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_handoff_limits import (
    DEFAULT_TASK_HANDOFF_LIMITS,
)
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_submission_error import (
    TaskSubmissionError,
)
from agent_team.domain.workflow.task_transition_error import (
    TaskTransitionError,
)
from agent_team.domain.workflow.task_verification_configuration_error import (
    TaskVerificationConfigurationError,
)
from agent_team.domain.workflow.task_verification_profiles import (
    is_valid_verification_contract,
)
from agent_team.domain.workflow.workflow_repository import WorkflowRepository
from agent_team.domain.workflow.workflow_validation_error import (
    WorkflowValidationError,
)

ACTIVE_TASK_STATUSES = frozenset(
    {
        TaskStatus.IN_PROGRESS,
        TaskStatus.VERIFICATION_PENDING,
    },
)
INITIAL_TASK_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.BLOCKED})
MODEL_STATUS_TRANSITIONS = {
    TaskStatus.PENDING: frozenset(
        {
            TaskStatus.IN_PROGRESS,
            TaskStatus.BLOCKED,
        },
    ),
    TaskStatus.BLOCKED: frozenset({TaskStatus.IN_PROGRESS}),
    TaskStatus.IN_PROGRESS: frozenset({TaskStatus.BLOCKED}),
}
SUBMITTABLE_ROLES = frozenset(
    {
        DevelopmentRole.BACKEND_DEVELOPER,
        DevelopmentRole.FRONTEND_DEVELOPER,
    },
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
        _validate_initial_task_status(task_status)
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
        """Apply a model-accessible task lifecycle transition."""
        task = self._require_task(task_id)
        task_status = _parse_enum(status, TaskStatus, "status")
        _validate_model_transition(task, task_status)
        if task_status is TaskStatus.IN_PROGRESS:
            updated_task = self.repository.claim_task_for_work(
                task_id=task_id,
                from_statuses=frozenset({task.status}),
                active_statuses=ACTIVE_TASK_STATUSES,
            )
            if updated_task is None:
                raise TaskActiveSlotConflictError(
                    "Another task for this feature and role is already "
                    "active.",
                )
            return updated_task
        updated_task = self.repository.transition_task_status(
            task_id=task_id,
            from_statuses=frozenset({task.status}),
            to_status=task_status,
        )
        if updated_task is None:
            raise TaskTransitionError("Task status changed before update.")
        return updated_task

    def submit_task_for_verification(
        self,
        draft: TaskHandoffDraft,
    ) -> TaskHandoff:
        """Persist a structured handoff and submit a task for verification."""
        task = self._require_task(draft.task_id)
        _validate_submission_task(task, draft)
        _validate_handoff(draft)
        handoff = self.repository.submit_task_handoff(
            draft=draft,
            from_status=TaskStatus.IN_PROGRESS,
            to_status=TaskStatus.VERIFICATION_PENDING,
        )
        if handoff is None:
            raise TaskSubmissionError(
                "Task must be in_progress before submission.",
            )
        return handoff

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


def _validate_model_transition(
    task: DevelopmentTask,
    to_status: TaskStatus,
) -> None:
    allowed = MODEL_STATUS_TRANSITIONS.get(task.status, frozenset())
    if to_status not in allowed:
        raise TaskTransitionError(
            f"Task cannot transition from {task.status.value} to "
            f"{to_status.value} through model-accessible tools.",
        )


def _validate_initial_task_status(status: TaskStatus) -> None:
    if status in INITIAL_TASK_STATUSES:
        return
    valid_values = ", ".join(
        task_status.value
        for task_status in TaskStatus
        if task_status in INITIAL_TASK_STATUSES
    )
    raise WorkflowValidationError(
        "Initial task status must be one of: "
        f"{valid_values}. Use submit_task_for_verification to enter "
        "verification_pending.",
    )


def _validate_submission_task(
    task: DevelopmentTask,
    draft: TaskHandoffDraft,
) -> None:
    if task.assigned_role not in SUBMITTABLE_ROLES:
        raise TaskSubmissionError(
            f"{task.assigned_role.value} tasks cannot be submitted by the "
            "developer harness.",
        )
    if draft.submitted_by is not task.assigned_role:
        raise TaskSubmissionError(
            "Submission role must match the assigned task role.",
        )
    if task.status is not TaskStatus.IN_PROGRESS:
        raise TaskSubmissionError(
            "Task must be in_progress before submission.",
        )
    if not is_valid_verification_contract(
        task.assigned_role,
        task.verification_contract,
    ):
        raise TaskVerificationConfigurationError(
            "Task verification configuration is missing or invalid.",
        )


def _validate_handoff(draft: TaskHandoffDraft) -> None:
    limits = DEFAULT_TASK_HANDOFF_LIMITS
    _require_limited_text(
        draft.attribution,
        "attribution",
        limits.item_chars,
    )
    _require_limited_text(
        draft.workspace_identity_hash,
        "workspace_identity_hash",
        limits.item_chars,
    )
    _require_limited_text(
        draft.implementation_summary,
        "implementation_summary",
        limits.implementation_summary_chars,
    )
    _require_limited_text(
        draft.next_action,
        "next_action",
        limits.next_action_chars,
    )
    _require_limited_text(
        draft.reuse_notes,
        "reuse_notes",
        limits.reuse_notes_chars,
    )
    _validate_limited_text(
        draft.limitations,
        "limitations",
        limits.limitations_chars,
    )
    _require_limited_items(
        draft.changed_paths,
        "changed_paths",
        limits.changed_path_count,
    )
    _validate_limited_items(
        draft.reused_symbols,
        "reused_symbols",
        limits.reused_symbol_count,
    )
    _validate_limited_items(
        draft.new_symbols,
        "new_symbols",
        limits.new_symbol_count,
    )
    _require_limited_items(
        draft.checks_attempted,
        "checks_attempted",
        limits.checks_attempted_count,
    )
    _validate_total_handoff_size(draft)


def _require_limited_text(
    value: str,
    field_name: str,
    max_chars: int,
) -> None:
    _require_text(value, field_name)
    _validate_limited_text(value, field_name, max_chars)


def _validate_limited_text(
    value: str,
    field_name: str,
    max_chars: int,
) -> None:
    if len(value) > max_chars:
        raise WorkflowValidationError(
            f"{field_name} must be at most {max_chars} characters.",
        )


def _require_limited_items(
    values: tuple[str, ...],
    field_name: str,
    max_count: int,
) -> None:
    if not values:
        raise WorkflowValidationError(f"{field_name} must not be empty.")
    _validate_limited_items(values, field_name, max_count)


def _validate_limited_items(
    values: tuple[str, ...],
    field_name: str,
    max_count: int,
) -> None:
    limits = DEFAULT_TASK_HANDOFF_LIMITS
    if len(values) > max_count:
        raise WorkflowValidationError(
            f"{field_name} must contain at most {max_count} items.",
        )
    normalized = tuple(value.strip() for value in values if value.strip())
    if len(normalized) != len(values):
        raise WorkflowValidationError(
            f"{field_name} must contain only non-blank values.",
        )
    for index, value in enumerate(values, start=1):
        if len(value) > limits.item_chars:
            raise WorkflowValidationError(
                f"{field_name} item {index} must be at most "
                f"{limits.item_chars} characters.",
            )
    if len(normalized) != len(set(normalized)):
        raise WorkflowValidationError(
            f"{field_name} must not contain duplicates.",
        )


def _validate_total_handoff_size(draft: TaskHandoffDraft) -> None:
    limits = DEFAULT_TASK_HANDOFF_LIMITS
    total_chars = sum(
        len(value)
        for value in (
            draft.attribution,
            draft.workspace_identity_hash,
            draft.implementation_summary,
            draft.reuse_notes,
            draft.limitations,
            draft.next_action,
        )
    ) + sum(
        len(item)
        for items in (
            draft.changed_paths,
            draft.reused_symbols,
            draft.new_symbols,
            draft.checks_attempted,
        )
        for item in items
    )
    if total_chars > limits.total_chars:
        raise WorkflowValidationError(
            "task handoff total accepted text must be at most "
            f"{limits.total_chars} characters.",
        )
