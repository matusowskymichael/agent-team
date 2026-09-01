"""Tests for Agent Skill Markdown parsing."""

import pytest

from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)
from agent_team.infrastructure.skills.skill_markdown_parser import (
    SkillMarkdownParser,
)


class TestSkillMarkdownParser:
    """SkillMarkdownParser behavior tests."""

    def test_parses_valid_skill_metadata_and_body(self) -> None:
        """Parse required and optional frontmatter fields safely."""
        parser = SkillMarkdownParser()

        metadata = parser.parse_metadata(
            directory_name="write-requirements-artifact",
            content_hash="hash",
            text=_skill_text(),
        )
        skill = parser.parse_skill(
            directory_name="write-requirements-artifact",
            content_hash="hash",
            text=_skill_text(),
        )

        assert metadata.name.value == "write-requirements-artifact"
        assert metadata.description.startswith("Use when")
        assert metadata.version == "0.1.0"
        assert metadata.allowed_tools == ("add_artifact",)
        assert skill.body == "Follow this procedure."

    def test_rejects_missing_frontmatter(self) -> None:
        """Require Agent Skills frontmatter."""
        with pytest.raises(InvalidAgentSkillError, match="frontmatter"):
            SkillMarkdownParser().parse_metadata(
                directory_name="bad-skill",
                content_hash="hash",
                text="No frontmatter.",
            )

    def test_rejects_missing_required_fields(self) -> None:
        """Require name and description fields."""
        text = "---\nname: bad-skill\n---\nBody."

        with pytest.raises(InvalidAgentSkillError, match="description"):
            SkillMarkdownParser().parse_metadata(
                directory_name="bad-skill",
                content_hash="hash",
                text=text,
            )

    def test_rejects_directory_name_mismatch(self) -> None:
        """Require directory name to match frontmatter name."""
        with pytest.raises(InvalidAgentSkillError, match="directory"):
            SkillMarkdownParser().parse_metadata(
                directory_name="other-skill",
                content_hash="hash",
                text=_skill_text(),
            )

    def test_rejects_executable_code_fences(self) -> None:
        """Disable script-like instructions in this slice."""
        text = _skill_text("```bash\necho nope\n```")

        with pytest.raises(InvalidAgentSkillError, match="script"):
            SkillMarkdownParser().parse_skill(
                directory_name="write-requirements-artifact",
                content_hash="hash",
                text=text,
            )

    def test_rejects_oversized_skill_body(self) -> None:
        """Enforce the configured body size limit."""
        text = _skill_text("x" * 13_000)

        with pytest.raises(InvalidAgentSkillError, match="too large"):
            SkillMarkdownParser().parse_skill(
                directory_name="write-requirements-artifact",
                content_hash="hash",
                text=text,
            )


def _skill_text(body: str = "Follow this procedure.") -> str:
    return (
        "---\n"
        "name: write-requirements-artifact\n"
        "description: Use when asked to add requirements.\n"
        "metadata:\n"
        "  version: 0.1.0\n"
        "allowed-tools:\n"
        "  - add_artifact\n"
        "---\n"
        f"{body}"
    )
