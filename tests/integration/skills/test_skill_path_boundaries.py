"""Validate canonical local skill paths against filesystem boundaries."""

from pathlib import Path

import pytest

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
from agent_team.infrastructure.skills.skill_path_policy import SkillPathPolicy


class TestSkillPathBoundaries:
    """Keep metadata discovery and resource loading inside reviewed paths."""

    def test_absent_root_has_no_discoverable_skills(
        self, tmp_path: Path
    ) -> None:
        """Treat an absent local knowledge root as an empty catalog."""
        assert SkillPathPolicy(tmp_path / "absent").skill_directories() == ()

    def test_root_files_do_not_become_skills(
        self, contained_skill_directory: Path
    ) -> None:
        """Ignore root files and canonicalize nested resource paths."""
        root = contained_skill_directory.parent
        (root / "README.md").write_text("Catalog.", encoding="utf-8")
        policy = SkillPathPolicy(root)
        assert policy.skill_directories() == (contained_skill_directory,)
        assert (
            policy.resource_file(
                contained_skill_directory, "references/notes.md"
            )
            == contained_skill_directory / "references/notes.md"
        )

    def test_hidden_catalog_entry_is_invalid(self, tmp_path: Path) -> None:
        """Reject hidden content before exposing skill metadata."""
        (tmp_path / ".hidden").mkdir()
        with pytest.raises(InvalidAgentSkillError, match="Hidden"):
            SkillPathPolicy(tmp_path).skill_directories()

    def test_regular_file_cannot_be_selected_as_skill(
        self, tmp_path: Path
    ) -> None:
        """Require a directory when resolving a skill name."""
        (tmp_path / "safe-skill").write_text("File.", encoding="utf-8")
        with pytest.raises(AgentSkillNotFoundError):
            SkillPathPolicy(tmp_path).skill_directory(
                AgentSkillName("safe-skill")
            )

    def test_missing_skill_file_is_invalid(self, tmp_path: Path) -> None:
        """Reject incomplete packages without the procedural entry point."""
        with pytest.raises(InvalidAgentSkillError, match=r"missing SKILL\.md"):
            SkillPathPolicy(tmp_path).skill_file(tmp_path)

    @pytest.mark.parametrize("path", (" ", "references/execute.py"))
    def test_resource_requests_cannot_be_blank_or_executable(
        self, contained_skill_directory: Path, path: str
    ) -> None:
        """Deny dangerous resource names before any filesystem read."""
        with pytest.raises(AgentSkillAccessDeniedError):
            SkillPathPolicy(contained_skill_directory.parent).resource_file(
                contained_skill_directory, path
            )

    def test_directory_cannot_be_loaded_as_resource(
        self, contained_skill_directory: Path
    ) -> None:
        """Resources must be contained regular files."""
        with pytest.raises(AgentSkillNotFoundError):
            SkillPathPolicy(contained_skill_directory.parent).resource_file(
                contained_skill_directory, "references"
            )

    @pytest.mark.parametrize(
        ("filename", "executable", "message"),
        (
            (".hidden", False, "Hidden"),
            ("password.md", False, "Secret-like"),
            ("procedure.md", True, "Executable"),
        ),
    )
    def test_package_selection_rejects_hidden_secret_or_executable_files(
        self,
        contained_skill_directory: Path,
        filename: str,
        executable: bool,
        message: str,
    ) -> None:
        """Nested files cannot smuggle scripts or sensitive local contents."""
        path = contained_skill_directory / "references" / filename
        path.write_text("Unreviewed content.", encoding="utf-8")
        if executable:
            path.chmod(0o700)
        with pytest.raises(InvalidAgentSkillError, match=message):
            SkillPathPolicy(contained_skill_directory.parent).skill_directory(
                AgentSkillName("safe-skill")
            )
