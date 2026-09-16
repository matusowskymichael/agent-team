"""Tests for workflow application service."""

from dataclasses import replace

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
from agent_team.domain.workflow.task_active_slot_conflict_error import (
    TaskActiveSlotConflictError,
)
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_handoff_limits import (
    DEFAULT_TASK_HANDOFF_LIMITS,
)
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_transition_error import (
    TaskTransitionError,
)
from agent_team.domain.workflow.task_verification_configuration_error import (
    TaskVerificationConfigurationError,
)
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
            status="in_progress",
        )

        assert feature.title == "Build MCP server"
        assert feature.description == "Store development work."
        assert feature.status == FeatureStatus.ANALYSIS
        assert service.get_feature(feature.id) == feature
        assert service.list_features(FeatureStatus.ANALYSIS) == [feature]
        assert artifact.kind == ArtifactKind.REQUIREMENTS
        assert service.list_artifacts(feature.id) == [artifact]
        assert task.assigned_role == DevelopmentRole.BACKEND_DEVELOPER
        assert updated_task.status == TaskStatus.IN_PROGRESS
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

    def test_create_task_rejects_verification_pending_status(self) -> None:
        """Reject direct creation in verification-only lifecycle states."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature(
            title="Feature",
            description="Description",
        )

        with pytest.raises(WorkflowValidationError, match="Initial task"):
            service.create_task(
                feature_id=feature.id,
                title="Task",
                description="Description",
                assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
                status=TaskStatus.VERIFICATION_PENDING,
            )

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

    @pytest.mark.parametrize(
        ("from_status", "to_status"),
        [
            (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),
            (TaskStatus.PENDING, TaskStatus.BLOCKED),
            (TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS),
            (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED),
        ],
    )
    def test_allows_model_accessible_transitions(
        self,
        from_status: TaskStatus,
        to_status: TaskStatus,
    ) -> None:
        """Allow only explicit model-accessible task transitions."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        task = repository.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
            from_status,
        )

        updated_task = service.update_task_status(task.id, to_status)

        assert updated_task.status is to_status

    @pytest.mark.parametrize(
        ("from_status", "to_status"),
        [
            (TaskStatus.PENDING, TaskStatus.COMPLETED),
            (TaskStatus.PENDING, TaskStatus.VERIFICATION_PENDING),
            (TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS),
            (TaskStatus.VERIFICATION_PENDING, TaskStatus.COMPLETED),
        ],
    )
    def test_rejects_forbidden_model_transitions(
        self,
        from_status: TaskStatus,
        to_status: TaskStatus,
    ) -> None:
        """Reject status transitions reserved for deterministic code."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        task = repository.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
            from_status,
        )

        with pytest.raises(TaskTransitionError):
            service.update_task_status(task.id, to_status)

    def test_active_same_role_task_blocks_another_claim(self) -> None:
        """Allow only one active task per feature and role."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        first_task = service.create_task(
            feature.id,
            "First",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        second_task = service.create_task(
            feature.id,
            "Second",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        service.update_task_status(first_task.id, TaskStatus.IN_PROGRESS)

        with pytest.raises(TaskActiveSlotConflictError):
            service.update_task_status(
                second_task.id,
                TaskStatus.IN_PROGRESS,
            )

    def test_submit_task_for_verification_persists_handoff(self) -> None:
        """Submit an in-progress implementation task for verification."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        task = service.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        service.update_task_status(task.id, TaskStatus.IN_PROGRESS)

        handoff = service.submit_task_for_verification(
            _handoff_draft(task.id),
        )

        assert handoff.task_id == task.id
        assert handoff.changed_paths == ("src/app.py",)
        submitted_task = repository.get_task(task.id)
        assert submitted_task is not None
        assert submitted_task.status is TaskStatus.VERIFICATION_PENDING

    def test_submit_requires_valid_verification_configuration(self) -> None:
        """Reject submission when the task has no trusted check contract."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        task = service.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        repository.tasks[task.id] = replace(
            task,
            status=TaskStatus.IN_PROGRESS,
            verification_contract=None,
        )

        with pytest.raises(TaskVerificationConfigurationError):
            service.submit_task_for_verification(_handoff_draft(task.id))

    def test_submit_accepts_handoff_at_exact_field_boundaries(self) -> None:
        """Accept values exactly at individual handoff field limits."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        task = service.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        service.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        limits = DEFAULT_TASK_HANDOFF_LIMITS

        handoff = service.submit_task_for_verification(
            replace(
                _handoff_draft(task.id),
                implementation_summary=(
                    "s" * limits.implementation_summary_chars
                ),
                reuse_notes="r" * limits.reuse_notes_chars,
                limitations="l" * limits.limitations_chars,
                next_action="n" * limits.next_action_chars,
                changed_paths=("p" * limits.item_chars,),
                reused_symbols=("r" * limits.item_chars,),
                new_symbols=("n" * limits.item_chars,),
                checks_attempted=("c" * limits.item_chars,),
            ),
        )

        assert len(handoff.implementation_summary) == (
            limits.implementation_summary_chars
        )

    def test_submit_rejects_handoff_over_text_boundary(self) -> None:
        """Reject handoff text beyond an individual field limit."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        task = service.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        service.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        limits = DEFAULT_TASK_HANDOFF_LIMITS
        oversized = replace(
            _handoff_draft(task.id),
            implementation_summary=(
                "s" * (limits.implementation_summary_chars + 1)
            ),
        )

        with pytest.raises(
            WorkflowValidationError,
            match="implementation_summary",
        ):
            service.submit_task_for_verification(oversized)

    def test_submit_rejects_oversized_handoff(self) -> None:
        """Bound persisted handoff fields before storage."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        task = service.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        service.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        limits = DEFAULT_TASK_HANDOFF_LIMITS
        oversized = replace(
            _handoff_draft(task.id),
            changed_paths=tuple(
                f"src/file_{index}.py"
                for index in range(limits.changed_path_count + 1)
            ),
        )

        with pytest.raises(WorkflowValidationError, match="changed_paths"):
            service.submit_task_for_verification(oversized)

    def test_submit_rejects_total_handoff_size_over_limit(self) -> None:
        """Reject collectively large handoffs inside individual limits."""
        repository = FakeWorkflowRepository()
        service = WorkflowService(repository=repository)
        feature = service.create_feature("Feature", "Description")
        task = service.create_task(
            feature.id,
            "Task",
            "Description",
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        service.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        limits = DEFAULT_TASK_HANDOFF_LIMITS
        oversized = replace(
            _handoff_draft(task.id),
            implementation_summary=("s" * limits.implementation_summary_chars),
            changed_paths=tuple(
                f"src/{index:02d}/" + ("p" * (limits.item_chars - 7))
                for index in range(limits.changed_path_count)
            ),
        )

        with pytest.raises(WorkflowValidationError, match="total accepted"):
            service.submit_task_for_verification(oversized)


def _handoff_draft(task_id: int) -> TaskHandoffDraft:
    return TaskHandoffDraft(
        task_id=task_id,
        agent_run_id=1,
        submitted_by=DevelopmentRole.BACKEND_DEVELOPER,
        attribution="agent:backend_developer",
        workspace_identity_hash="workspace-hash",
        implementation_summary="Patched the assigned backend behavior.",
        changed_paths=("src/app.py",),
        reused_symbols=("ExistingService",),
        new_symbols=("NewHandler",),
        reuse_notes="Existing service was reused for persistence.",
        checks_attempted=("backend",),
        limitations="none",
        next_action="run deterministic verification",
    )
