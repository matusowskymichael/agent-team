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

    @pytest.mark.parametrize(
        "text",
        (
            "---\nname: safe-skill\nBody without closing frontmatter.",
            "---\nname: safe-skill\ndescription: "
            + "x" * 4_001
            + "\n---\nBody.",
            "---\n name: safe-skill\n---\nBody.",
            "---\nname safe-skill\n---\nBody.",
            "---\n: safe-skill\n---\nBody.",
            "---\nname: []\ndescription: Procedure.\n---\nBody.",
            "---\nname: safe-skill\ndescription: Procedure.\n"
            "allowed-tools: read_file\n---\nBody.",
            "---\nname: safe-skill\ndescription: Procedure.\n---\n  ",
        ),
        ids=(
            "unclosed-frontmatter",
            "oversized-frontmatter",
            "unsupported-indent",
            "missing-separator",
            "missing-key",
            "nontext-name",
            "nonlist-tool-metadata",
            "empty-body",
        ),
    )
    def test_rejects_malformed_metadata_before_exposing_instructions(
        self, text: str
    ) -> None:
        """Fail closed for malformed or oversized procedural packages."""
        with pytest.raises(InvalidAgentSkillError):
            SkillMarkdownParser().parse_skill("safe-skill", "hash", text)

    @pytest.mark.parametrize(
        ("optional_fields", "version", "tools"),
        (
            ("", None, ()),
            ("metadata: plain-text\nallowed-tools: []\n", None, ()),
            ("metadata:\n  owner: local\n", None, ()),
            ("metadata:\n  version: ''\n", None, ()),
            (
                "metadata:\n  version: '0.2.0'\n"
                "allowed-tools: ['read_file', , '', \"find_symbol\"]\n",
                "0.2.0",
                ("read_file", "find_symbol"),
            ),
        ),
        ids=(
            "absent",
            "empty-tools",
            "no-version",
            "blank-version",
            "quoted-inline-list",
        ),
    )
    def test_optional_metadata_keeps_portable_defaults(
        self,
        optional_fields: str,
        version: str | None,
        tools: tuple[str, ...],
    ) -> None:
        """Accept comments and optional metadata without creating authority."""
        text = (
            "---\n\n# Reviewed local procedure.\nname: safe-skill\n"
            "description: Procedure.\n" + optional_fields + "---\nBody."
        )
        metadata = SkillMarkdownParser().parse_metadata(
            "safe-skill", "hash", text
        )
        assert metadata.version == version
        assert metadata.allowed_tools == tools


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
