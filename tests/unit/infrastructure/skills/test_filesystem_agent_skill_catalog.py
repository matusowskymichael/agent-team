"""Tests for filesystem-backed Agent Skill catalog."""

from pathlib import Path
from typing import cast

import pytest

from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)
from agent_team.infrastructure.skills.filesystem_agent_skill_catalog import (
    FilesystemAgentSkillCatalog,
)
from agent_team.infrastructure.skills.skill_path_policy import (
    SkillPathPolicy,
)


class TestFilesystemAgentSkillCatalog:
    """FilesystemAgentSkillCatalog behavior tests."""

    def test_discovers_metadata_without_exposing_body(
        self,
        tmp_path: Path,
    ) -> None:
        """Return portable metadata during discovery."""
        _write_skill(tmp_path, "write-requirements-artifact")

        metadata = FilesystemAgentSkillCatalog(tmp_path).list_metadata()

        assert len(metadata) == 1
        assert metadata[0].name.value == "write-requirements-artifact"
        assert metadata[0].version == "0.1.0"
        assert not hasattr(metadata[0], "body")

    def test_load_skill_returns_body_after_explicit_request(
        self,
        tmp_path: Path,
    ) -> None:
        """Progressively load the skill body by name."""
        _write_skill(tmp_path, "write-requirements-artifact")

        skill = FilesystemAgentSkillCatalog(tmp_path).load_skill(
            AgentSkillName("write-requirements-artifact"),
        )

        assert "Body for write-requirements-artifact." in skill.body
        assert skill.metadata.content_hash

    def test_reads_contained_resource(
        self,
        tmp_path: Path,
    ) -> None:
        """Read a non-secret resource within a selected skill."""
        skill_dir = _write_skill(tmp_path, "write-requirements-artifact")
        (skill_dir / "notes.md").write_text("notes")

        content, content_hash = FilesystemAgentSkillCatalog(
            tmp_path,
        ).read_resource(
            AgentSkillName("write-requirements-artifact"),
            "notes.md",
        )

        assert content == "notes"
        assert len(content_hash) == 64

    def test_rejects_directory_name_mismatch(self, tmp_path: Path) -> None:
        """Reject skills whose frontmatter name differs from directory."""
        _write_skill(
            tmp_path,
            "write-requirements-artifact",
            frontmatter_name="other-skill",
        )

        with pytest.raises(InvalidAgentSkillError, match="directory"):
            FilesystemAgentSkillCatalog(tmp_path).list_metadata()

    def test_rejects_duplicate_skill_names(self, tmp_path: Path) -> None:
        """Detect duplicate metadata names defensively."""
        directory = _write_skill(tmp_path, "write-requirements-artifact")
        fake_policy = _DuplicatePathPolicy(tmp_path, directory)

        with pytest.raises(InvalidAgentSkillError, match="Duplicate"):
            FilesystemAgentSkillCatalog(
                tmp_path,
                path_policy=cast("SkillPathPolicy", fake_policy),
            ).list_metadata()


class _DuplicatePathPolicy:
    def __init__(self, root: Path, directory: Path) -> None:
        self.root = root
        self.directory = directory

    def skill_directories(self) -> tuple[Path, ...]:
        return (self.directory, self.directory)

    def skill_file(self, _directory: Path) -> Path:
        return self.directory / "SKILL.md"


def _write_skill(
    root: Path,
    name: str,
    frontmatter_name: str | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir()
    metadata_name = name if frontmatter_name is None else frontmatter_name
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {metadata_name}\n"
        f"description: Description for {name}.\n"
        "metadata:\n"
        "  version: 0.1.0\n"
        "---\n"
        f"Body for {name}."
    )
    return skill_dir
