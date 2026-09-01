"""Tests for Agent Skill filesystem path policy."""

from pathlib import Path

import pytest

from agent_team.domain.skills.agent_skill_access_denied_error import (
    AgentSkillAccessDeniedError,
)
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)
from agent_team.infrastructure.skills.skill_path_policy import (
    SkillPathPolicy,
)


class TestSkillPathPolicy:
    """SkillPathPolicy behavior tests."""

    def test_rejects_absolute_resource_path(self, tmp_path: Path) -> None:
        """Reject absolute paths before resolving resources."""
        skill_dir = _skill_dir(tmp_path)

        with pytest.raises(AgentSkillAccessDeniedError):
            SkillPathPolicy(tmp_path).resource_file(skill_dir, "/etc/passwd")

    def test_rejects_traversal_resource_path(self, tmp_path: Path) -> None:
        """Reject traversal paths."""
        skill_dir = _skill_dir(tmp_path)

        with pytest.raises(AgentSkillAccessDeniedError):
            SkillPathPolicy(tmp_path).resource_file(skill_dir, "../other.md")

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """Reject resources resolving outside the selected skill."""
        skill_dir = _skill_dir(tmp_path)
        outside = tmp_path / "outside.md"
        outside.write_text("secret")
        (skill_dir / "outside.md").symlink_to(outside)

        with pytest.raises(AgentSkillAccessDeniedError):
            SkillPathPolicy(tmp_path).resource_file(skill_dir, "outside.md")

    def test_rejects_hidden_resource(self, tmp_path: Path) -> None:
        """Reject hidden resource names."""
        skill_dir = _skill_dir(tmp_path)
        (skill_dir / ".hidden").write_text("hidden")

        with pytest.raises(AgentSkillAccessDeniedError):
            SkillPathPolicy(tmp_path).resource_file(skill_dir, ".hidden")

    def test_rejects_secret_like_resource(self, tmp_path: Path) -> None:
        """Reject files whose names look secret-bearing."""
        skill_dir = _skill_dir(tmp_path)
        (skill_dir / "api_key.txt").write_text("secret")

        with pytest.raises(AgentSkillAccessDeniedError):
            SkillPathPolicy(tmp_path).resource_file(skill_dir, "api_key.txt")

    def test_rejects_oversized_resource(self, tmp_path: Path) -> None:
        """Reject resource files above the configured size limit."""
        skill_dir = _skill_dir(tmp_path)
        (skill_dir / "notes.md").write_text("x" * 10)

        with pytest.raises(InvalidAgentSkillError, match="size"):
            SkillPathPolicy(tmp_path, max_resource_bytes=1).resource_file(
                skill_dir,
                "notes.md",
            )

    def test_rejects_scripts_in_skill_directory(self, tmp_path: Path) -> None:
        """Disable skills containing script files."""
        skill_dir = _skill_dir(tmp_path)
        (skill_dir / "run.sh").write_text("echo nope")

        with pytest.raises(InvalidAgentSkillError, match="scripts"):
            SkillPathPolicy(tmp_path).skill_directory(
                AgentSkillName("safe-skill"),
            )


def _skill_dir(root: Path) -> Path:
    skill_dir = root / "safe-skill"
    skill_dir.mkdir()
    return skill_dir
