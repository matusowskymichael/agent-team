"""Tests for the restricted local workspace executor."""

import subprocess
from pathlib import Path

import pytest

from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    MAX_LIST_FILES,
    MAX_READ_CHARS,
    MAX_SEARCH_MATCHES,
    MAX_SYMBOL_DEFINITIONS,
    LocalWorkspaceExecutor,
)


class TestLocalWorkspaceExecutor:
    """Local workspace executor security and behavior tests."""

    def test_rejects_absolute_paths(self, tmp_path: Path) -> None:
        """Reject absolute model-supplied paths."""
        executor = LocalWorkspaceExecutor(tmp_path)

        with pytest.raises(WorkspaceAccessDeniedError):
            executor.read_file(str(tmp_path / "backend/auth.py"))

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Reject traversal outside the trusted root."""
        executor = LocalWorkspaceExecutor(tmp_path)

        with pytest.raises(WorkspaceAccessDeniedError):
            executor.read_file("../outside.py")

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """Reject symlinks that resolve outside the trusted root."""
        outside = tmp_path.parent / "outside-secret.txt"
        outside.write_text("secret", encoding="utf-8")
        link = tmp_path / "backend" / "outside.txt"
        link.parent.mkdir()
        link.symlink_to(outside)
        executor = LocalWorkspaceExecutor(tmp_path)

        with pytest.raises(WorkspaceAccessDeniedError):
            executor.read_file("backend/outside.txt")

    def test_excludes_ignored_directories(self, tmp_path: Path) -> None:
        """Hide repository metadata, virtualenvs, and cache output."""
        _write(tmp_path, "backend/auth.py", "def login():\n    return True\n")
        _write(tmp_path, ".git/config", "private")
        _write(tmp_path, ".venv/token.py", "private")
        _write(tmp_path, "__pycache__/mod.pyc", "private")
        executor = LocalWorkspaceExecutor(tmp_path)

        listing = executor.list_files("")

        assert listing.files == ("backend/auth.py",)

    def test_excludes_secret_files(self, tmp_path: Path) -> None:
        """Hide common local secret files."""
        _write(tmp_path, "backend/auth.py", "safe")
        _write(tmp_path, ".env", "TOKEN=secret")
        _write(tmp_path, "backend/private.key", "secret")
        executor = LocalWorkspaceExecutor(tmp_path)

        listing = executor.list_files("")

        assert listing.files == ("backend/auth.py",)

    def test_custom_ignored_paths_hide_nested_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Apply configured workspace ignore patterns."""
        _write(tmp_path, "backend/public.py", "safe")
        _write(tmp_path, "backend/private/secret.py", "private")
        executor = LocalWorkspaceExecutor(
            tmp_path,
            ignored_paths=("backend/private",),
        )

        listing = executor.list_files("")

        assert listing.files == ("backend/public.py",)

    def test_file_listing_is_bounded(self, tmp_path: Path) -> None:
        """Bound large file listings."""
        for index in range(MAX_LIST_FILES + 5):
            _write(tmp_path, f"backend/file_{index}.py", "")
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.list_files("")

        assert len(result.files) == MAX_LIST_FILES
        assert result.truncated is True

    def test_reads_are_bounded(self, tmp_path: Path) -> None:
        """Return bounded content with a full-content hash."""
        content = "a" * (MAX_READ_CHARS + 10)
        _write(tmp_path, "backend/large.py", content)
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.read_file("backend/large.py")

        assert result.content == "a" * MAX_READ_CHARS
        assert result.truncated is True
        assert len(result.content_hash) == 64

    def test_search_results_are_bounded(self, tmp_path: Path) -> None:
        """Return bounded literal search results."""
        for index in range(MAX_SEARCH_MATCHES + 5):
            _write(tmp_path, f"backend/file_{index}.py", "needle\n")
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.search_code("needle")

        assert len(result.matches) == MAX_SEARCH_MATCHES
        assert result.truncated is True

    def test_blank_search_query_is_rejected(self, tmp_path: Path) -> None:
        """Reject blank search input before execution."""
        executor = LocalWorkspaceExecutor(tmp_path)

        with pytest.raises(WorkspaceAccessDeniedError, match="must not"):
            executor.search_code("   ")

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

    def test_python_search_fallback_skips_unreadable_binary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use the Python fallback and skip non-UTF-8 files."""
        _write(tmp_path, "backend/auth.py", "needle\n")
        binary_path = tmp_path / "backend" / "binary.py"
        binary_path.write_bytes(b"\xff")
        monkeypatch.setattr(
            "agent_team.infrastructure.workspace.local_workspace_executor."
            "shutil.which",
            _missing_executable,
        )
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.search_code("needle")

        assert len(result.matches) == 1
        assert result.matches[0].path == "backend/auth.py"

    def test_rg_failure_falls_back_to_python_search(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fall back to Python search when rg fails unexpectedly."""
        _write(tmp_path, "backend/auth.py", "needle\n")

        def fake_run(
            _command: tuple[str, ...],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=(),
                returncode=2,
                stdout="",
                stderr="rg error",
            )

        monkeypatch.setattr(
            "agent_team.infrastructure.workspace.local_workspace_executor."
            "shutil.which",
            _rg_executable,
        )
        monkeypatch.setattr(
            "agent_team.infrastructure.workspace.local_workspace_executor."
            "subprocess.run",
            fake_run,
        )
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.search_code("needle")

        assert result.matches[0].path == "backend/auth.py"

    def test_malformed_rg_lines_are_ignored(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ignore malformed or unauthorized rg output lines."""
        _write(tmp_path, "backend/auth.py", "needle\n")

        def fake_run(
            _command: tuple[str, ...],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout=(
                    "broken\n"
                    "backend/auth.py:not-int:needle\n"
                    "backend/auth.py:1:needle\n"
                    "backend/missing.py:1:needle\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(
            "agent_team.infrastructure.workspace.local_workspace_executor."
            "shutil.which",
            _rg_executable,
        )
        monkeypatch.setattr(
            "agent_team.infrastructure.workspace.local_workspace_executor."
            "subprocess.run",
            fake_run,
        )
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.search_code("needle")

        assert [match.path for match in result.matches] == ["backend/auth.py"]

    def test_applies_patch_inside_workspace(self, tmp_path: Path) -> None:
        """Apply an exact replacement inside the trusted root."""
        _write(tmp_path, "backend/auth.py", "def login():\n    return False\n")
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.apply_patch(
            "backend/auth.py",
            "return False",
            "return True",
        )

        assert result.applied is True
        assert (tmp_path / "backend/auth.py").read_text() == (
            "def login():\n    return True\n"
        )
        assert result.before_hash != result.after_hash

    def test_creates_new_file_inside_workspace(self, tmp_path: Path) -> None:
        """Create a new file when old_text is empty."""
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.apply_patch(
            "backend/new_module.py",
            "",
            "VALUE = 1\n",
        )

        assert result.applied is True
        assert (tmp_path / "backend/new_module.py").read_text() == (
            "VALUE = 1\n"
        )

    def test_patch_reports_missing_old_text(self, tmp_path: Path) -> None:
        """Return a non-applied patch result when old text is absent."""
        _write(tmp_path, "backend/auth.py", "return False\n")
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.apply_patch("backend/auth.py", "missing", "new")

        assert result.applied is False
        assert result.message == "old_text was not found"
        assert (tmp_path / "backend/auth.py").read_text() == "return False\n"

    def test_patch_rejects_replacement_in_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject replacing text in a file that does not exist."""
        executor = LocalWorkspaceExecutor(tmp_path)

        with pytest.raises(WorkspaceAccessDeniedError, match="missing file"):
            executor.apply_patch("backend/missing.py", "old", "new")

    def test_runs_allowlisted_check(self, tmp_path: Path) -> None:
        """Run configured project checks without shell access."""
        executor = LocalWorkspaceExecutor(
            tmp_path,
            check_commands={
                "backend": ("python", "-c", "print('ok')"),
            },
        )

        result = executor.run_check("backend")

        assert result.exit_code == 0
        assert result.stdout_excerpt == "ok"
        assert result.timed_out is False

    def test_check_timeout_returns_bounded_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Convert check timeouts into check results."""

        def fake_run(
            command: tuple[str, ...],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(
                cmd=command,
                timeout=1,
                output=b"partial output",
                stderr=None,
            )

        monkeypatch.setattr(
            "agent_team.infrastructure.workspace.local_workspace_executor."
            "subprocess.run",
            fake_run,
        )
        executor = LocalWorkspaceExecutor(
            tmp_path,
            check_commands={"backend": ("python", "-c", "print('ok')")},
        )

        result = executor.run_check("backend")

        assert result.exit_code == 124
        assert result.stdout_excerpt == "partial output"
        assert result.stderr_excerpt == ""
        assert result.timed_out is True

    def test_rejects_unconfigured_check(self, tmp_path: Path) -> None:
        """Reject arbitrary command names."""
        executor = LocalWorkspaceExecutor(tmp_path, check_commands={})

        with pytest.raises(WorkspaceAccessDeniedError):
            executor.run_check("rm -rf .")

    def test_rejects_destructive_check(self, tmp_path: Path) -> None:
        """Reject destructive commands even if misconfigured."""
        executor = LocalWorkspaceExecutor(
            tmp_path,
            check_commands={"bad": ("rm", "-rf", ".")},
        )

        with pytest.raises(WorkspaceAccessDeniedError):
            executor.run_check("bad")

    @pytest.mark.parametrize(
        "command",
        [
            (),
            ("bash", "-c", "echo no"),
        ],
    )
    def test_rejects_unsafe_check_commands(
        self,
        tmp_path: Path,
        command: tuple[str, ...],
    ) -> None:
        """Reject empty or unsupported check command definitions."""
        executor = LocalWorkspaceExecutor(
            tmp_path,
            check_commands={"bad": command},
        )

        with pytest.raises(WorkspaceAccessDeniedError):
            executor.run_check("bad")

    @pytest.mark.parametrize(
        ("operation", "argument"),
        [
            ("list", "backend/missing"),
            ("list", "backend/auth.py"),
            ("list", ".git"),
            ("read", "backend/missing.py"),
            ("read", "backend"),
            ("read", ".env"),
        ],
    )
    def test_rejects_invalid_paths(
        self,
        tmp_path: Path,
        operation: str,
        argument: str,
    ) -> None:
        """Reject missing, wrong-kind, or excluded workspace paths."""
        _write(tmp_path, "backend/auth.py", "safe")
        _write(tmp_path, ".git/config", "private")
        _write(tmp_path, ".env", "TOKEN=secret")
        executor = LocalWorkspaceExecutor(tmp_path)

        with pytest.raises(WorkspaceAccessDeniedError):
            if operation == "list":
                executor.list_files(argument)
            else:
                executor.read_file(argument)

    def test_empty_workspace_is_usable(self, tmp_path: Path) -> None:
        """Allow listing an empty trusted workspace."""
        executor = LocalWorkspaceExecutor(tmp_path)

        result = executor.list_files("")

        assert result.files == ()
        assert result.truncated is False


def _write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _missing_executable(_name: str) -> str | None:
    return None


def _rg_executable(_name: str) -> str | None:
    return "/usr/bin/rg"
