"""Restricted local workspace executor."""

import fnmatch
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from agent_team.application.audit.audit_sanitizer import (
    hash_text,
    sanitize_text,
)
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.code_search_match import CodeSearchMatch
from agent_team.domain.workspace.code_search_result import CodeSearchResult
from agent_team.domain.workspace.patch_application_result import (
    PatchApplicationResult,
)
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_file_content import (
    WorkspaceFileContent,
)
from agent_team.domain.workspace.workspace_file_listing import (
    WorkspaceFileListing,
)

MAX_LIST_FILES = 250
MAX_SEARCH_MATCHES = 80
MAX_READ_CHARS = 20_000
MAX_CHECK_OUTPUT_CHARS = 4_000
CHECK_TIMEOUT_SECONDS = 60.0
DEFAULT_IGNORED_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".pyright",
        "node_modules",
        "dist",
        "build",
        "coverage",
        "htmlcov",
    },
)
DEFAULT_SECRET_PATTERNS = frozenset(
    {
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "id_rsa",
        "id_dsa",
        "*secret*",
        "*token*",
        "*credential*",
    },
)
ALLOWED_CHECK_EXECUTABLES = frozenset(
    {
        "python",
        "python3",
        "uv",
        "pytest",
        "ruff",
        "pyright",
    },
)
DESTRUCTIVE_CHECK_TOKENS = frozenset(
    {
        "rm",
        "rmdir",
        "unlink",
        "shred",
        "mv",
        "git",
        "curl",
        "wget",
        "ssh",
        "scp",
    },
)


def _default_check_commands() -> dict[str, tuple[str, ...]]:
    return {
        "backend": ("uv", "run", "pytest"),
        "frontend": ("uv", "run", "pytest"),
        "pytest": ("uv", "run", "pytest"),
        "ruff": ("uv", "run", "ruff", "check", "."),
        "pyright": ("uv", "run", "pyright"),
    }


def _empty_patterns() -> tuple[str, ...]:
    return ()


@dataclass(frozen=True, slots=True)
class LocalWorkspaceExecutor:
    """Execute bounded filesystem operations inside one workspace root."""

    root: Path
    check_commands: Mapping[str, tuple[str, ...]] = field(
        default_factory=_default_check_commands,
    )
    ignored_paths: tuple[str, ...] = field(default_factory=_empty_patterns)

    def list_files(self, directory: str = "") -> WorkspaceFileListing:
        """Return visible files under a workspace-relative directory."""
        start = self._resolve_directory(directory)
        files: list[str] = []
        truncated = False
        for path in _walk_files(start):
            if self._is_excluded(path):
                continue
            files.append(_relative_path(self._root(), path))
            if len(files) >= MAX_LIST_FILES:
                truncated = True
                break
        return WorkspaceFileListing(
            files=tuple(sorted(files)),
            truncated=truncated,
        )

    def search_code(self, query: str) -> CodeSearchResult:
        """Search visible workspace files for a literal query."""
        clean_query = query.strip()
        if not clean_query:
            raise WorkspaceAccessDeniedError("Search query must not be blank.")
        matches = (
            self._search_with_rg(clean_query)
            if shutil.which("rg") is not None
            else self._search_with_python(clean_query)
        )
        return CodeSearchResult(
            query_hash=hash_text(clean_query),
            matches=tuple(matches[:MAX_SEARCH_MATCHES]),
            truncated=len(matches) > MAX_SEARCH_MATCHES,
        )

    def read_file(self, path: str) -> WorkspaceFileContent:
        """Read bounded UTF-8 content from a visible workspace file."""
        target = self._resolve_file(path, must_exist=True)
        content = target.read_text(encoding="utf-8")
        truncated = len(content) > MAX_READ_CHARS
        visible_content = content[:MAX_READ_CHARS] if truncated else content
        return WorkspaceFileContent(
            path=_relative_path(self._root(), target),
            content=visible_content,
            content_hash=hash_text(content),
            truncated=truncated,
        )

    def apply_patch(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ) -> PatchApplicationResult:
        """Apply an exact text replacement or create a new file."""
        target = self._resolve_file(path, must_exist=False)
        if target.exists():
            before = target.read_text(encoding="utf-8")
            before_hash = hash_text(before)
            if old_text not in before:
                return PatchApplicationResult(
                    path=_relative_path(self._root(), target),
                    applied=False,
                    before_hash=before_hash,
                    after_hash=before_hash,
                    line_count_delta=0,
                    message="old_text was not found",
                )
            after = before.replace(old_text, new_text, 1)
        else:
            if old_text:
                raise WorkspaceAccessDeniedError(
                    "Cannot replace text in a missing file.",
                )
            before = ""
            before_hash = None
            after = new_text
            target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(after, encoding="utf-8")
        return PatchApplicationResult(
            path=_relative_path(self._root(), target),
            applied=True,
            before_hash=before_hash,
            after_hash=hash_text(after),
            line_count_delta=_line_count(after) - _line_count(before),
            message="patch applied",
        )

    def run_check(self, name: str) -> CheckRunResult:
        """Run one configured project check without shell expansion."""
        command = self.check_commands.get(name)
        if command is None:
            raise WorkspaceAccessDeniedError(
                f"Workspace check {name} is not configured.",
            )
        _validate_check_command(command)
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                cwd=self._root(),
                check=False,
                capture_output=True,
                text=True,
                timeout=CHECK_TIMEOUT_SECONDS,
            )
            return CheckRunResult(
                name=name,
                exit_code=completed.returncode,
                stdout_excerpt=sanitize_text(
                    completed.stdout[:MAX_CHECK_OUTPUT_CHARS],
                ),
                stderr_excerpt=sanitize_text(
                    completed.stderr[:MAX_CHECK_OUTPUT_CHARS],
                ),
                timed_out=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout = _optional_output(error.stdout)
            stderr = _optional_output(error.stderr)
            return CheckRunResult(
                name=name,
                exit_code=124,
                stdout_excerpt=sanitize_text(stdout[:MAX_CHECK_OUTPUT_CHARS]),
                stderr_excerpt=sanitize_text(stderr[:MAX_CHECK_OUTPUT_CHARS]),
                timed_out=True,
            )

    def _search_with_rg(self, query: str) -> list[CodeSearchMatch]:
        command = (
            "rg",
            "-F",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--",
            query,
            ".",
        )
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=self._root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT_SECONDS,
        )
        if completed.returncode not in {0, 1}:
            return self._search_with_python(query)
        return [
            match
            for match in (
                self._parse_rg_line(line)
                for line in completed.stdout.splitlines()
            )
            if match is not None
        ]

    def _parse_rg_line(self, line: str) -> CodeSearchMatch | None:
        path_text, separator, remainder = line.partition(":")
        if not separator:
            return None
        line_number_text, separator, excerpt = remainder.partition(":")
        if not separator:
            return None
        try:
            line_number = int(line_number_text)
        except ValueError:
            return None
        try:
            path = self._resolve_file(path_text, must_exist=True)
        except WorkspaceAccessDeniedError:
            return None
        return CodeSearchMatch(
            path=_relative_path(self._root(), path),
            line_number=line_number,
            line_excerpt=sanitize_text(excerpt),
        )

    def _search_with_python(self, query: str) -> list[CodeSearchMatch]:
        matches: list[CodeSearchMatch] = []
        for path in _walk_files(self._root()):
            if self._is_excluded(path):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(
                        CodeSearchMatch(
                            path=_relative_path(self._root(), path),
                            line_number=line_number,
                            line_excerpt=sanitize_text(line),
                        ),
                    )
        return matches

    def _resolve_directory(self, path: str) -> Path:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise WorkspaceAccessDeniedError("Workspace directory not found.")
        if not resolved.is_dir():
            raise WorkspaceAccessDeniedError(
                "Workspace path is not a directory.",
            )
        if self._is_excluded(resolved):
            raise WorkspaceAccessDeniedError("Workspace path is excluded.")
        return resolved

    def _resolve_file(self, path: str, must_exist: bool) -> Path:
        resolved = self._resolve(path)
        if must_exist and not resolved.is_file():
            raise WorkspaceAccessDeniedError("Workspace file not found.")
        if resolved.exists() and not resolved.is_file():
            raise WorkspaceAccessDeniedError("Workspace path is not a file.")
        if self._is_excluded(resolved):
            raise WorkspaceAccessDeniedError("Workspace path is excluded.")
        return resolved

    def _resolve(self, path: str) -> Path:
        root = self._root()
        if not path.strip():
            return root
        requested = Path(path.replace("\\", "/"))
        if requested.is_absolute() or ".." in requested.parts:
            raise WorkspaceAccessDeniedError(
                "Workspace path is outside the root.",
            )
        resolved = (root / requested).resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise WorkspaceAccessDeniedError(
                "Workspace path is outside the root.",
            )
        return resolved

    def _root(self) -> Path:
        root = self.root.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _is_excluded(self, path: Path) -> bool:
        relative = _relative_path(self._root(), path)
        if relative == ".":
            return False
        parts = Path(relative).parts
        if any(part in DEFAULT_IGNORED_NAMES for part in parts):
            return True
        if any(
            _matches_pattern(relative, pattern)
            for pattern in self.ignored_paths
        ):
            return True
        name = path.name.lower()
        return any(
            fnmatch.fnmatch(name, pattern)
            for pattern in DEFAULT_SECRET_PATTERNS
        )


def _walk_files(start: Path) -> Iterable[Path]:
    for directory, dir_names, file_names in os.walk(start, followlinks=False):
        dir_names[:] = [
            name
            for name in dir_names
            if name not in DEFAULT_IGNORED_NAMES
            and not _is_hidden_cache_directory(name)
        ]
        current = Path(directory)
        for file_name in file_names:
            yield current / file_name


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _matches_pattern(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern) or path.startswith(
        f"{pattern.rstrip('/')}/",
    )


def _is_hidden_cache_directory(name: str) -> bool:
    return name.startswith(".") and name.endswith("_cache")


def _line_count(value: str) -> int:
    if not value:
        return 0
    return len(value.splitlines())


def _validate_check_command(command: tuple[str, ...]) -> None:
    if not command:
        raise WorkspaceAccessDeniedError("Workspace check command is empty.")
    executable = Path(command[0]).name
    if (
        executable not in ALLOWED_CHECK_EXECUTABLES
        and not executable.startswith("python")
    ):
        raise WorkspaceAccessDeniedError(
            f"Workspace check executable {executable} is not allowed.",
        )
    for token in command:
        if token in DESTRUCTIVE_CHECK_TOKENS:
            raise WorkspaceAccessDeniedError(
                f"Workspace check token {token} is not allowed.",
            )


def _optional_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
