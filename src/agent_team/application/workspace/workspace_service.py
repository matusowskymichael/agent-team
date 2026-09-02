"""Application service for restricted workspace operations."""

from dataclasses import dataclass
from pathlib import PurePosixPath

from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.workflow.workflow_repository import WorkflowRepository
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.code_search_result import CodeSearchResult
from agent_team.domain.workspace.patch_application_result import (
    PatchApplicationResult,
)
from agent_team.domain.workspace.symbol_search_result import SymbolSearchResult
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_binding_error import (
    WorkspaceBindingError,
)
from agent_team.domain.workspace.workspace_executor import WorkspaceExecutor
from agent_team.domain.workspace.workspace_file_content import (
    WorkspaceFileContent,
)
from agent_team.domain.workspace.workspace_file_listing import (
    WorkspaceFileListing,
)
from agent_team.domain.workspace.workspace_tool_name import WorkspaceToolName


@dataclass(frozen=True, slots=True)
class WorkspaceService:
    """Authorize and run feature-task-bound workspace operations."""

    repository: WorkflowRepository
    executor: WorkspaceExecutor

    def authorize(
        self,
        profile: AgentProfile,
        task: AgentTask,
        tool: WorkspaceToolName,
        path: str = "",
        mutation: bool = False,
    ) -> None:
        """Authorize a workspace operation without executing it."""
        self._authorize(profile, task, tool, path, mutation)

    def list_files(
        self,
        profile: AgentProfile,
        task: AgentTask,
        directory: str = "",
    ) -> WorkspaceFileListing:
        """List visible workspace files after binding checks."""
        self._authorize(
            profile,
            task,
            WorkspaceToolName.LIST_FILES,
            directory,
            mutation=False,
        )
        return self.executor.list_files(directory)

    def search_code(
        self,
        profile: AgentProfile,
        task: AgentTask,
        query: str,
    ) -> CodeSearchResult:
        """Search visible workspace files after binding checks."""
        self._authorize(
            profile,
            task,
            WorkspaceToolName.SEARCH_CODE,
            "",
            mutation=False,
        )
        if not query.strip():
            raise WorkspaceAccessDeniedError("Search query must not be blank.")
        return self.executor.search_code(query)

    def find_symbol(
        self,
        profile: AgentProfile,
        task: AgentTask,
        name: str,
    ) -> SymbolSearchResult:
        """Locate exact source definitions after binding checks."""
        self._authorize(
            profile,
            task,
            WorkspaceToolName.FIND_SYMBOL,
            "",
            mutation=False,
        )
        if not name.strip():
            raise WorkspaceAccessDeniedError("Symbol name must not be blank.")
        return self.executor.find_symbol(name)

    def read_file(
        self,
        profile: AgentProfile,
        task: AgentTask,
        path: str,
    ) -> WorkspaceFileContent:
        """Read a visible workspace file after binding checks."""
        self._authorize(
            profile,
            task,
            WorkspaceToolName.READ_FILE,
            path,
            mutation=False,
        )
        return self.executor.read_file(path)

    def apply_patch(
        self,
        profile: AgentProfile,
        task: AgentTask,
        path: str,
        old_text: str,
        new_text: str,
    ) -> PatchApplicationResult:
        """Apply a bounded patch after binding and path checks."""
        self._authorize(
            profile,
            task,
            WorkspaceToolName.APPLY_PATCH,
            path,
            mutation=True,
        )
        return self.executor.apply_patch(path, old_text, new_text)

    def run_check(
        self,
        profile: AgentProfile,
        task: AgentTask,
        name: str,
    ) -> CheckRunResult:
        """Run an allowlisted project check after binding checks."""
        self._authorize(
            profile,
            task,
            WorkspaceToolName.RUN_CHECK,
            "",
            mutation=False,
        )
        if name not in profile.allowed_workspace_checks:
            raise WorkspaceAccessDeniedError(
                f"The {profile.role.value} role cannot run check {name}.",
            )
        return self.executor.run_check(name)

    def _authorize(
        self,
        profile: AgentProfile,
        task: AgentTask,
        tool: WorkspaceToolName,
        path: str,
        mutation: bool,
    ) -> None:
        if tool not in profile.allowed_workspace_tools:
            raise WorkspaceAccessDeniedError(
                f"The {profile.role.value} role cannot call {tool.value}.",
            )
        self._require_binding(profile, task, mutation)
        if path:
            _validate_path_prefix(profile, path)

    def _require_binding(
        self,
        profile: AgentProfile,
        task: AgentTask,
        mutation: bool,
    ) -> None:
        if task.feature_id is None:
            raise WorkspaceBindingError(
                "Workspace tools require a trusted feature binding.",
            )
        if task.task_id is None:
            raise WorkspaceBindingError(
                "Workspace tools require a trusted task binding.",
            )
        if task.workspace_root is None:
            raise WorkspaceBindingError(
                "Workspace tools require a trusted workspace root.",
            )
        development_task = self.repository.get_task(task.task_id)
        if development_task is None:
            raise WorkspaceBindingError(
                f"Development task {task.task_id} was not found.",
            )
        if development_task.feature_id != task.feature_id:
            raise WorkspaceBindingError(
                "Development task does not belong to the bound feature.",
            )
        if development_task.assigned_role is not profile.role:
            raise WorkspaceAccessDeniedError(
                f"The {profile.role.value} role cannot use task "
                f"{task.task_id}.",
            )
        if mutation and development_task.assigned_role is not task.role:
            raise WorkspaceAccessDeniedError(
                "Workspace mutation requires matching trusted task role.",
            )


def _validate_path_prefix(profile: AgentProfile, path: str) -> None:
    normalized = _normalized_path(path)
    if not profile.allowed_workspace_path_prefixes:
        raise WorkspaceAccessDeniedError(
            f"The {profile.role.value} role has no writable path prefixes.",
        )
    if any(
        _matches_prefix(normalized, prefix)
        for prefix in profile.allowed_workspace_path_prefixes
    ):
        return
    raise WorkspaceAccessDeniedError(
        f"The {profile.role.value} role cannot access workspace path "
        f"{normalized}.",
    )


def _normalized_path(path: str) -> str:
    pure_path = PurePosixPath(path.replace("\\", "/"))
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise WorkspaceAccessDeniedError("Workspace path is outside the root.")
    normalized = pure_path.as_posix().strip("/")
    if not normalized:
        raise WorkspaceAccessDeniedError("Workspace path must not be blank.")
    return normalized


def _matches_prefix(path: str, prefix: str) -> bool:
    clean_prefix = prefix.strip("/")
    if not clean_prefix:
        return True
    return path == clean_prefix or path.startswith(f"{clean_prefix}/")
