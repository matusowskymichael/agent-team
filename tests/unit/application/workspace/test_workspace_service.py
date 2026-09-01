"""Tests for restricted workspace application service."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.workspace.workspace_service import WorkspaceService
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.code_search_result import CodeSearchResult
from agent_team.domain.workspace.patch_application_result import (
    PatchApplicationResult,
)
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_binding_error import (
    WorkspaceBindingError,
)
from agent_team.domain.workspace.workspace_file_content import (
    WorkspaceFileContent,
)
from agent_team.domain.workspace.workspace_file_listing import (
    WorkspaceFileListing,
)
from agent_team.domain.workspace.workspace_tool_name import WorkspaceToolName
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


def _check_call_store() -> list[str]:
    return []


@dataclass(slots=True)
class _FakeWorkspaceExecutor:
    patch_calls: int = 0
    check_calls: list[str] = field(default_factory=_check_call_store)

    def list_files(self, directory: str = "") -> WorkspaceFileListing:
        return WorkspaceFileListing(
            files=(directory or "backend/auth.py",),
            truncated=False,
        )

    def search_code(self, query: str) -> CodeSearchResult:
        return CodeSearchResult(query_hash=query, matches=(), truncated=False)

    def read_file(self, path: str) -> WorkspaceFileContent:
        return WorkspaceFileContent(
            path=path,
            content="content",
            content_hash="hash",
            truncated=False,
        )

    def apply_patch(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ) -> PatchApplicationResult:
        self.patch_calls += 1
        return PatchApplicationResult(
            path=path,
            applied=True,
            before_hash="before",
            after_hash="after",
            line_count_delta=len(new_text.splitlines())
            - len(old_text.splitlines()),
            message="patch applied",
        )

    def run_check(self, name: str) -> CheckRunResult:
        self.check_calls.append(name)
        return CheckRunResult(
            name=name,
            exit_code=0,
            stdout_excerpt="ok",
            stderr_excerpt="",
            timed_out=False,
        )


class TestWorkspaceService:
    """Workspace service authorization tests."""

    def test_allowed_backend_patch_delegates_to_executor(self) -> None:
        """Allow an assigned backend task to patch backend code."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        executor = _FakeWorkspaceExecutor()
        service = WorkspaceService(repository=repository, executor=executor)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        result = service.apply_patch(
            profile,
            _task(DevelopmentRole.BACKEND_DEVELOPER),
            "backend/auth.py",
            "old",
            "new",
        )

        assert result.applied is True
        assert executor.patch_calls == 1

    def test_read_only_operations_delegate_after_binding(self) -> None:
        """Delegate read-only workspace operations after trusted binding."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        executor = _FakeWorkspaceExecutor()
        service = WorkspaceService(repository=repository, executor=executor)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        task = _task(DevelopmentRole.BACKEND_DEVELOPER)

        listing = service.list_files(profile, task, "backend")
        search = service.search_code(profile, task, "AuthService")
        content = service.read_file(profile, task, "backend/auth.py")
        check = service.run_check(profile, task, "backend")

        assert listing.files == ("backend",)
        assert search.query_hash == "AuthService"
        assert content.path == "backend/auth.py"
        assert check.name == "backend"
        assert executor.check_calls == ["backend"]

    def test_missing_workspace_binding_is_denied(self) -> None:
        """Require a trusted workspace root before workspace operations."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        executor = _FakeWorkspaceExecutor()
        service = WorkspaceService(repository=repository, executor=executor)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        with pytest.raises(WorkspaceBindingError):
            service.list_files(
                profile,
                AgentTask(
                    prompt="List files.",
                    role=DevelopmentRole.BACKEND_DEVELOPER,
                    feature_id=1,
                    task_id=1,
                ),
            )

    @pytest.mark.parametrize(
        ("task", "message"),
        [
            (
                AgentTask(
                    prompt="List files.",
                    role=DevelopmentRole.BACKEND_DEVELOPER,
                    task_id=1,
                    workspace_root=Path("workspace"),
                ),
                "feature binding",
            ),
            (
                AgentTask(
                    prompt="List files.",
                    role=DevelopmentRole.BACKEND_DEVELOPER,
                    feature_id=1,
                    workspace_root=Path("workspace"),
                ),
                "task binding",
            ),
        ],
    )
    def test_missing_trusted_bindings_are_denied(
        self,
        task: AgentTask,
        message: str,
    ) -> None:
        """Reject workspace use without required trusted bindings."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        service = WorkspaceService(
            repository=repository,
            executor=_FakeWorkspaceExecutor(),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        with pytest.raises(WorkspaceBindingError, match=message):
            service.list_files(profile, task)

    def test_wrong_task_assignment_is_denied_before_patch(self) -> None:
        """Deny code mutation when the task is assigned to another role."""
        repository = _repository_with_task(DevelopmentRole.FRONTEND_DEVELOPER)
        executor = _FakeWorkspaceExecutor()
        service = WorkspaceService(repository=repository, executor=executor)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        with pytest.raises(WorkspaceAccessDeniedError):
            service.apply_patch(
                profile,
                _task(DevelopmentRole.BACKEND_DEVELOPER),
                "backend/auth.py",
                "old",
                "new",
            )

        assert executor.patch_calls == 0

    def test_non_workspace_role_cannot_call_workspace_tools(self) -> None:
        """Deny workspace tools not listed by the active role profile."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        service = WorkspaceService(
            repository=repository,
            executor=_FakeWorkspaceExecutor(),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BUSINESS_ANALYST,
        )

        with pytest.raises(WorkspaceAccessDeniedError, match="cannot call"):
            service.list_files(
                profile,
                _task(DevelopmentRole.BUSINESS_ANALYST),
            )

    def test_missing_task_record_is_a_binding_error(self) -> None:
        """Reject a trusted task ID that no longer exists."""
        service = WorkspaceService(
            repository=FakeWorkflowRepository(),
            executor=_FakeWorkspaceExecutor(),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        with pytest.raises(WorkspaceBindingError, match="was not found"):
            service.list_files(
                profile,
                _task(DevelopmentRole.BACKEND_DEVELOPER),
            )

    def test_task_feature_mismatch_is_a_binding_error(self) -> None:
        """Reject task IDs that belong to a different feature."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        service = WorkspaceService(
            repository=repository,
            executor=_FakeWorkspaceExecutor(),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        task = AgentTask(
            prompt="List files.",
            role=DevelopmentRole.BACKEND_DEVELOPER,
            feature_id=2,
            task_id=1,
            workspace_root=Path("workspace"),
        )

        with pytest.raises(WorkspaceBindingError, match="bound feature"):
            service.list_files(profile, task)

    def test_mutation_requires_task_role_to_match_runtime_role(self) -> None:
        """Deny mutation when prompt role differs from profile role."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        executor = _FakeWorkspaceExecutor()
        service = WorkspaceService(repository=repository, executor=executor)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        task = _task(DevelopmentRole.FRONTEND_DEVELOPER)

        with pytest.raises(WorkspaceAccessDeniedError, match="task role"):
            service.apply_patch(profile, task, "backend/auth.py", "old", "new")

        assert executor.patch_calls == 0

    def test_unauthorized_path_is_denied_before_patch(self) -> None:
        """Deny backend mutation of frontend-only paths."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        executor = _FakeWorkspaceExecutor()
        service = WorkspaceService(repository=repository, executor=executor)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        with pytest.raises(WorkspaceAccessDeniedError):
            service.apply_patch(
                profile,
                _task(DevelopmentRole.BACKEND_DEVELOPER),
                "frontend/LoginForm.tsx",
                "old",
                "new",
            )

        assert executor.patch_calls == 0

    @pytest.mark.parametrize(
        "path",
        ["../outside.py", "/outside.py"],
    )
    def test_invalid_model_supplied_paths_are_denied(self, path: str) -> None:
        """Reject blank, absolute, and traversal paths."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        service = WorkspaceService(
            repository=repository,
            executor=_FakeWorkspaceExecutor(),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        with pytest.raises(WorkspaceAccessDeniedError):
            service.read_file(
                profile,
                _task(DevelopmentRole.BACKEND_DEVELOPER),
                path,
            )

    def test_profile_without_path_prefixes_cannot_access_files(self) -> None:
        """Reject file access when no path prefixes are configured."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        service = WorkspaceService(
            repository=repository,
            executor=_FakeWorkspaceExecutor(),
        )
        profile = _custom_profile(
            role=DevelopmentRole.BACKEND_DEVELOPER,
            prefixes=frozenset(),
        )

        with pytest.raises(WorkspaceAccessDeniedError, match="no writable"):
            service.read_file(
                profile,
                _task(DevelopmentRole.BACKEND_DEVELOPER),
                "backend/auth.py",
            )

    def test_empty_path_prefix_allows_bounded_access(self) -> None:
        """Allow a custom profile that deliberately grants the whole root."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        service = WorkspaceService(
            repository=repository,
            executor=_FakeWorkspaceExecutor(),
        )
        profile = _custom_profile(
            role=DevelopmentRole.BACKEND_DEVELOPER,
            prefixes=frozenset({""}),
        )

        result = service.read_file(
            profile,
            _task(DevelopmentRole.BACKEND_DEVELOPER),
            "docs/readme.md",
        )

        assert result.path == "docs/readme.md"

    def test_unconfigured_check_is_denied(self) -> None:
        """Allow only checks configured in the immutable role profile."""
        repository = _repository_with_task(DevelopmentRole.FRONTEND_DEVELOPER)
        executor = _FakeWorkspaceExecutor()
        service = WorkspaceService(repository=repository, executor=executor)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.FRONTEND_DEVELOPER,
        )

        with pytest.raises(WorkspaceAccessDeniedError):
            service.run_check(
                profile,
                _task(DevelopmentRole.FRONTEND_DEVELOPER),
                "pyright",
            )

        assert executor.check_calls == []

    def test_blank_search_query_is_denied_before_executor(self) -> None:
        """Reject blank search queries in application validation."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        executor = _FakeWorkspaceExecutor()
        service = WorkspaceService(repository=repository, executor=executor)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )

        with pytest.raises(WorkspaceAccessDeniedError, match="must not"):
            service.search_code(
                profile,
                _task(DevelopmentRole.BACKEND_DEVELOPER),
                "   ",
            )


def _repository_with_task(
    assigned_role: DevelopmentRole,
) -> FakeWorkflowRepository:
    repository = FakeWorkflowRepository()
    feature = repository.create_feature(
        title="Feature",
        description="Description.",
        status=FeatureStatus.DRAFT,
    )
    repository.create_task(
        feature_id=feature.id,
        title="Task",
        description="Task description.",
        assigned_role=assigned_role,
        status=TaskStatus.PENDING,
    )
    return repository


def _task(role: DevelopmentRole) -> AgentTask:
    return AgentTask(
        prompt="Patch code.",
        role=role,
        feature_id=1,
        task_id=1,
        workspace_root=Path("workspace"),
    )


def _custom_profile(
    role: DevelopmentRole,
    prefixes: frozenset[str],
) -> AgentProfile:
    return AgentProfile(
        role=role,
        instructions="Test profile.",
        allowed_tools=frozenset(WorkflowToolName),
        run_limits=AgentRunLimits(max_turns=6),
        allowed_workspace_tools=frozenset(WorkspaceToolName),
        allowed_workspace_path_prefixes=prefixes,
        allowed_workspace_checks=frozenset({"backend"}),
    )
