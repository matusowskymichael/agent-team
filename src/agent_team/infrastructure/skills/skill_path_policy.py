"""Filesystem containment policy for local Agent Skill files."""

import os
from dataclasses import dataclass
from pathlib import Path

from agent_team.domain.skills.agent_skill_access_denied_error import (
    AgentSkillAccessDeniedError,
)
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.domain.skills.agent_skill_not_found_error import (
    AgentSkillNotFoundError,
)
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)

MAX_SKILL_FILE_BYTES = 32_000
MAX_SKILL_RESOURCE_BYTES = 32_000
SKILL_FILE_NAME = "SKILL.md"
SCRIPT_SUFFIXES = frozenset(
    {
        ".bat",
        ".cmd",
        ".js",
        ".pl",
        ".ps1",
        ".py",
        ".rb",
        ".sh",
        ".ts",
    },
)
SECRET_NAME_TERMS = frozenset(
    {
        ".env",
        "api_key",
        "authorization",
        "credential",
        "id_rsa",
        "password",
        "private_key",
        "secret",
        "token",
    },
)


@dataclass(frozen=True, slots=True)
class SkillPathPolicy:
    """Validate skill paths before filesystem reads."""

    root: Path
    max_skill_file_bytes: int = MAX_SKILL_FILE_BYTES
    max_resource_bytes: int = MAX_SKILL_RESOURCE_BYTES

    def skill_directories(self) -> tuple[Path, ...]:
        """Return validated immediate child skill directories."""
        root = self._root()
        if not root.exists():
            return ()
        directories: list[Path] = []
        for path in sorted(root.iterdir()):
            if path.name.startswith("."):
                raise InvalidAgentSkillError("Hidden skill directories fail.")
            if path.is_dir():
                directories.append(
                    self.skill_directory(AgentSkillName(path.name))
                )
        return tuple(directories)

    def skill_directory(self, name: AgentSkillName) -> Path:
        """Return the canonical directory for one skill name."""
        root = self._root()
        candidate = (root / name.value).resolve(strict=False)
        if not _is_relative_to(candidate, root) or not candidate.exists():
            raise AgentSkillNotFoundError(f"Skill {name.value} was not found.")
        if not candidate.is_dir():
            raise AgentSkillNotFoundError(f"Skill {name.value} was not found.")
        self._reject_scripts(candidate)
        return candidate

    def skill_file(self, directory: Path) -> Path:
        """Return a validated SKILL.md path for a skill directory."""
        path = (directory / SKILL_FILE_NAME).resolve(strict=False)
        if not path.exists():
            raise InvalidAgentSkillError(
                "Skill directory is missing SKILL.md.",
            )
        _reject_oversized_file(path, self.max_skill_file_bytes)
        return path

    def resource_file(
        self,
        directory: Path,
        relative_path: str,
    ) -> Path:
        """Return a contained non-secret resource file path."""
        raw_path = Path(relative_path)
        if raw_path.is_absolute() or not relative_path.strip():
            raise AgentSkillAccessDeniedError("Skill resource path denied.")
        if ".." in raw_path.parts:
            raise AgentSkillAccessDeniedError("Skill resource path denied.")
        if any(part.startswith(".") for part in raw_path.parts):
            raise AgentSkillAccessDeniedError("Hidden skill resources denied.")
        if _is_secret_like(raw_path):
            raise AgentSkillAccessDeniedError("Secret-like resource denied.")
        if _is_script(raw_path):
            raise AgentSkillAccessDeniedError("Skill scripts are disabled.")

        skill_root = directory.resolve(strict=True)
        candidate = (skill_root / raw_path).resolve(strict=True)
        if not _is_relative_to(candidate, skill_root):
            raise AgentSkillAccessDeniedError("Skill resource path denied.")
        if not candidate.is_file():
            raise AgentSkillNotFoundError("Skill resource was not found.")
        _reject_oversized_file(candidate, self.max_resource_bytes)
        return candidate

    def _root(self) -> Path:
        return self.root.resolve(strict=False)

    def _reject_scripts(self, directory: Path) -> None:
        for path in directory.rglob("*"):
            if path.name.startswith("."):
                raise InvalidAgentSkillError("Hidden skill files fail.")
            if not path.is_file():
                continue
            if _is_secret_like(path):
                raise InvalidAgentSkillError("Secret-like skill files fail.")
            if _is_script(path) or os.access(path, os.X_OK):
                raise InvalidAgentSkillError(
                    "Executable scripts are disabled for Agent Skills.",
                )


def _reject_oversized_file(path: Path, limit: int) -> None:
    if path.stat().st_size > limit:
        raise InvalidAgentSkillError("Skill file exceeds size limits.")


def _is_script(path: Path) -> bool:
    return path.suffix.lower() in SCRIPT_SUFFIXES


def _is_secret_like(path: Path) -> bool:
    clean = path.name.lower()
    return any(term in clean for term in SECRET_NAME_TERMS)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
