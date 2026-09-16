"""Integration tests for workspace mutation lifecycle locking."""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import cast

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.sessions.workspace_identity import (
    workspace_identity_hash,
)
from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.application.workspace.workspace_service import (
    WorkspaceService,
)
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.code_search_result import CodeSearchResult
from agent_team.domain.workspace.patch_application_result import (
    PatchApplicationResult,
)
from agent_team.domain.workspace.symbol_search_result import SymbolSearchResult
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_executor import WorkspaceExecutor
from agent_team.domain.workspace.workspace_file_content import (
    WorkspaceFileContent,
)
from agent_team.domain.workspace.workspace_file_listing import (
    WorkspaceFileListing,
)
from agent_team.infrastructure.persistence.sqlite.workflow.sqlite_workflow_repository import (  # noqa: E501
    SQLiteWorkflowRepository,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)


@dataclass(slots=True)
class _BlockingPatchExecutor:
    root: Path
    entered: Event
    release: Event

    def list_files(self, directory: str = "") -> WorkspaceFileListing:
        raise AssertionError(f"unexpected list_files {directory}")

    def search_code(self, query: str) -> CodeSearchResult:
        raise AssertionError(f"unexpected search_code {query}")

    def find_symbol(self, name: str) -> SymbolSearchResult:
        raise AssertionError(f"unexpected find_symbol {name}")

    def read_file(self, path: str) -> WorkspaceFileContent:
        raise AssertionError(f"unexpected read_file {path}")

    def apply_patch(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ) -> PatchApplicationResult:
        """Block while SQLite holds the task status mutation lock."""
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("patch release was not signaled")
        target = self.root / path
        before = target.read_text(encoding="utf-8")
        after = before.replace(old_text, new_text, 1)
        target.write_text(after, encoding="utf-8")
        return PatchApplicationResult(
            path=path,
            applied=True,
            before_hash="before",
            after_hash="after",
            line_count_delta=0,
            message="patch applied",
        )

    def run_check(self, name: str) -> CheckRunResult:
        raise AssertionError(f"unexpected run_check {name}")


class TestWorkspaceMutationLockIntegration:
    """Workspace mutation status lock integration tests."""

    def test_submission_winning_race_blocks_patch(
        self,
        tmp_path: Path,
    ) -> None:
        """Deny a patch that waits behind a verification transition."""
        database_path = tmp_path / "workflow.db"
        workspace_root = _workspace(tmp_path)
        repository, _workflow, task_id = _seed_in_progress_task(database_path)
        errors: list[BaseException] = []
        thread = Thread(
            target=lambda: _capture_patch_error(
                repository,
                task_id,
                workspace_root,
                errors,
            ),
        )
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE development_tasks
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    TaskStatus.VERIFICATION_PENDING.value,
                    datetime.now(UTC).isoformat(),
                    task_id,
                ),
            )
            thread.start()
            sleep(0.1)
        finally:
            connection.commit()
            connection.close()
        thread.join(timeout=5.0)

        assert not thread.is_alive()
        assert isinstance(errors[0], WorkspaceAccessDeniedError)
        assert "in_progress" in str(errors[0])
        assert (workspace_root / "backend/auth.py").read_text() == "old\n"
        task = repository.get_task(task_id)
        assert task is not None
        assert task.status is TaskStatus.VERIFICATION_PENDING

    def test_patch_winning_race_serializes_before_submission(
        self,
        tmp_path: Path,
    ) -> None:
        """Let an in-progress patch finish before submission transitions."""
        database_path = tmp_path / "workflow.db"
        workspace_root = _workspace(tmp_path)
        repository, workflow, task_id = _seed_in_progress_task(database_path)
        entered = Event()
        release = Event()
        executor = _BlockingPatchExecutor(workspace_root, entered, release)
        service = WorkspaceService(
            repository=repository,
            executor=cast("WorkspaceExecutor", executor),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        patch_errors: list[BaseException] = []
        submit_errors: list[BaseException] = []

        patch_thread = Thread(
            target=lambda: _capture(
                patch_errors,
                service.apply_patch,
                profile,
                _agent_task(task_id, workspace_root),
                "backend/auth.py",
                "old",
                "new",
            ),
        )
        patch_thread.start()
        assert entered.wait(timeout=5.0)

        submit_thread = Thread(
            target=lambda: _capture(
                submit_errors,
                workflow.submit_task_for_verification,
                _handoff(task_id, workspace_root),
            ),
        )
        submit_thread.start()
        sleep(0.1)
        assert submit_thread.is_alive()

        release.set()
        patch_thread.join(timeout=5.0)
        submit_thread.join(timeout=5.0)

        assert not patch_thread.is_alive()
        assert not submit_thread.is_alive()
        assert patch_errors == []
        assert submit_errors == []
        assert (workspace_root / "backend/auth.py").read_text() == "new\n"
        handoff = repository.latest_task_handoff(task_id)
        task = repository.get_task(task_id)
        assert handoff is not None
        assert task is not None
        assert task.status is TaskStatus.VERIFICATION_PENDING


def _seed_in_progress_task(
    database_path: Path,
) -> tuple[SQLiteWorkflowRepository, WorkflowService, int]:
    repository = SQLiteWorkflowRepository(database_path)
    workflow = WorkflowService(repository)
    feature = workflow.create_feature(
        "Feature",
        "Description.",
        FeatureStatus.DRAFT,
    )
    task = workflow.create_task(
        feature.id,
        "Task",
        "Description.",
        DevelopmentRole.BACKEND_DEVELOPER,
    )
    workflow.update_task_status(task.id, TaskStatus.IN_PROGRESS)
    return repository, workflow, task.id


def _workspace(tmp_path: Path) -> Path:
    workspace_root = tmp_path / "workspace"
    target = workspace_root / "backend/auth.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\n", encoding="utf-8")
    return workspace_root


def _capture_patch_error(
    repository: SQLiteWorkflowRepository,
    task_id: int,
    workspace_root: Path,
    errors: list[BaseException],
) -> None:
    service = WorkspaceService(
        repository=repository,
        executor=LocalWorkspaceExecutor(workspace_root),
    )
    profile = AgentProfileCatalog().get_profile(
        DevelopmentRole.BACKEND_DEVELOPER,
    )
    try:
        service.apply_patch(
            profile,
            _agent_task(task_id, workspace_root),
            "backend/auth.py",
            "old",
            "new",
        )
    except BaseException as error:
        errors.append(error)


def _capture(
    errors: list[BaseException],
    function: Callable[..., object],
    *arguments: object,
) -> None:
    try:
        function(*arguments)
    except BaseException as error:
        errors.append(error)


def _agent_task(task_id: int, workspace_root: Path) -> AgentTask:
    return AgentTask(
        prompt="Patch backend code.",
        role=DevelopmentRole.BACKEND_DEVELOPER,
        feature_id=1,
        task_id=task_id,
        workspace_root=workspace_root,
    )


def _handoff(task_id: int, workspace_root: Path) -> TaskHandoffDraft:
    return TaskHandoffDraft(
        task_id=task_id,
        agent_run_id=1,
        submitted_by=DevelopmentRole.BACKEND_DEVELOPER,
        attribution="agent:backend_developer",
        workspace_identity_hash=workspace_identity_hash(workspace_root),
        implementation_summary="Patched backend behavior.",
        changed_paths=("backend/auth.py",),
        reused_symbols=("AuthService",),
        new_symbols=(),
        reuse_notes="Existing auth module was reused.",
        checks_attempted=("backend",),
        limitations="none",
        next_action="run deterministic verification",
    )
