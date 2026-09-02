"""Filesystem integration tests for workspace symbol discovery."""

from pathlib import Path

import pytest

from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    MAX_SYMBOL_DEFINITIONS,
    LocalWorkspaceExecutor,
)


class TestLocalWorkspaceExecutorSymbolDiscovery:
    """Exercise symbol discovery against current filesystem contents."""

    def test_finds_python_class_function_and_qualified_method(
        self,
        tmp_path: Path,
    ) -> None:
        """Use Python AST nodes rather than textual symbol mentions."""
        _write(
            tmp_path,
            "backend/auth.py",
            (
                "class AuthService:\n"
                "    async def logout(self, token: str) -> bool:\n"
                "        return bool(token)\n\n"
                "def build_service() -> AuthService:\n"
                "    return AuthService()\n"
            ),
        )
        _write(
            tmp_path,
            "tests/test_auth.py",
            "# AuthService.logout is exercised here.\n",
        )
        executor = LocalWorkspaceExecutor(tmp_path)

        class_result = executor.find_symbol("AuthService")
        method_result = executor.find_symbol("AuthService.logout")
        function_result = executor.find_symbol("build_service")

        assert [item.kind for item in class_result.definitions] == ["class"]
        assert [item.qualified_name for item in method_result.definitions] == [
            "AuthService.logout",
        ]
        assert [item.kind for item in function_result.definitions] == [
            "function",
        ]
        assert all(
            item.path != "tests/test_auth.py"
            for item in method_result.definitions
        )

    def test_finds_javascript_typescript_symbols(self, tmp_path: Path) -> None:
        """Locate common frontend class, method, and function declarations."""
        _write(
            tmp_path,
            "frontend/AccountMenu.tsx",
            (
                "export class AccountMenu {\n"
                "  logout(): void {\n"
                "    return\n"
                "  }\n"
                "}\n"
                "export const LoginForm = () => null\n"
                "export async function loadProfile() { return null }\n"
            ),
        )
        executor = LocalWorkspaceExecutor(tmp_path)

        method = executor.find_symbol("AccountMenu.logout")
        component = executor.find_symbol("LoginForm")
        function = executor.find_symbol("loadProfile")

        assert method.definitions[0].kind == "method"
        assert component.definitions[0].kind == "function"
        assert function.definitions[0].kind == "function"

    def test_ignores_calls_inside_javascript_class_methods(
        self,
        tmp_path: Path,
    ) -> None:
        """Recognize methods only at direct class-body depth."""
        _write(
            tmp_path,
            "frontend/AccountMenu.ts",
            (
                "export class AccountMenu {\n"
                "  logout(): void {\n"
                "    revokeToken()\n"
                "    if (shouldRefresh()) {\n"
                "      refreshSession()\n"
                "    }\n"
                "  }\n"
                "\n"
                "  open(): void {}\n"
                "}\n"
            ),
        )
        executor = LocalWorkspaceExecutor(tmp_path)

        logout = executor.find_symbol("AccountMenu.logout")
        open_menu = executor.find_symbol("AccountMenu.open")
        revoke_token = executor.find_symbol("AccountMenu.revokeToken")
        refresh_session = executor.find_symbol("AccountMenu.refreshSession")

        assert [item.qualified_name for item in logout.definitions] == [
            "AccountMenu.logout",
        ]
        assert [item.qualified_name for item in open_menu.definitions] == [
            "AccountMenu.open",
        ]
        assert revoke_token.definitions == ()
        assert refresh_session.definitions == ()

    def test_symbol_search_reflects_current_workspace_contents(
        self,
        tmp_path: Path,
    ) -> None:
        """Rescan files on every call instead of serving stale results."""
        _write(tmp_path, "backend/service.py", "VALUE = 1\n")
        executor = LocalWorkspaceExecutor(tmp_path)

        before = executor.find_symbol("refresh")
        _write(
            tmp_path,
            "backend/service.py",
            "def refresh() -> None:\n    pass\n",
        )
        after_add = executor.find_symbol("refresh")
        _write(tmp_path, "backend/service.py", "VALUE = 2\n")
        after_remove = executor.find_symbol("refresh")

        assert before.definitions == ()
        assert len(after_add.definitions) == 1
        assert after_remove.definitions == ()

    def test_symbol_results_are_bounded(self, tmp_path: Path) -> None:
        """Bound exact symbol results across a large workspace."""
        for index in range(MAX_SYMBOL_DEFINITIONS + 5):
            _write(
                tmp_path,
                f"backend/service_{index}.py",
                "def shared_name() -> None:\n    pass\n",
            )
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.find_symbol("shared_name")

        assert len(result.definitions) == MAX_SYMBOL_DEFINITIONS
        assert result.truncated is True

    def test_blank_symbol_name_is_rejected(self, tmp_path: Path) -> None:
        """Reject blank exact-symbol queries."""
        executor = LocalWorkspaceExecutor(tmp_path)

        with pytest.raises(WorkspaceAccessDeniedError, match="must not"):
            executor.find_symbol("   ")

    def test_malformed_python_is_ignored_during_symbol_search(
        self,
        tmp_path: Path,
    ) -> None:
        """Skip incomplete Python files while finding other definitions."""
        _write(tmp_path, "backend/broken.py", "def incomplete(\n")
        _write(
            tmp_path,
            "backend/healthy.py",
            "def healthy() -> None:\n    pass\n",
        )
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.find_symbol("healthy")

        assert result.definitions[0].path == "backend/healthy.py"


def _write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
