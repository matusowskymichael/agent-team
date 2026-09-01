"""Workspace executor protocol."""

from typing import Protocol

from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.code_search_result import CodeSearchResult
from agent_team.domain.workspace.patch_application_result import (
    PatchApplicationResult,
)
from agent_team.domain.workspace.workspace_file_content import (
    WorkspaceFileContent,
)
from agent_team.domain.workspace.workspace_file_listing import (
    WorkspaceFileListing,
)


class WorkspaceExecutor(Protocol):
    """Port for restricted operations against one trusted workspace root."""

    def list_files(self, directory: str = "") -> WorkspaceFileListing:
        """Return visible files under a workspace-relative directory."""
        ...

    def search_code(self, query: str) -> CodeSearchResult:
        """Search visible workspace files for a literal query."""
        ...

    def read_file(self, path: str) -> WorkspaceFileContent:
        """Read bounded content from one visible workspace file."""
        ...

    def apply_patch(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ) -> PatchApplicationResult:
        """Apply an exact text replacement inside the workspace."""
        ...

    def run_check(self, name: str) -> CheckRunResult:
        """Run one configured workspace check by trusted name."""
        ...
