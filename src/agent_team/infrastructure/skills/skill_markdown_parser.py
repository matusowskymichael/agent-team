"""Parser for script-free Agent Skills SKILL.md files."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from agent_team.domain.skills.agent_skill import AgentSkill
from agent_team.domain.skills.agent_skill_metadata import AgentSkillMetadata
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)

MAX_SKILL_FRONTMATTER_CHARS = 4_000
MAX_SKILL_BODY_CHARS = 12_000
MIN_QUOTED_STRING_LENGTH = 2
EXECUTABLE_FENCE_PATTERN = re.compile(
    r"^```(?:bash|sh|shell|python|py|javascript|js|typescript|ts|"
    r"powershell|ps1|ruby|perl)\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class SkillMarkdownParser:
    """Parse the portable script-free Agent Skills Markdown format."""

    def parse_metadata(
        self,
        directory_name: str,
        content_hash: str,
        text: str,
    ) -> AgentSkillMetadata:
        """Parse frontmatter without returning instruction body content."""
        frontmatter, body = _split_skill_text(text)
        metadata = _parse_frontmatter(frontmatter)
        _validate_body(body)
        return _metadata_from_mapping(directory_name, content_hash, metadata)

    def parse_skill(
        self,
        directory_name: str,
        content_hash: str,
        text: str,
    ) -> AgentSkill:
        """Parse metadata and instruction body from a SKILL.md file."""
        frontmatter, body = _split_skill_text(text)
        metadata = _parse_frontmatter(frontmatter)
        _validate_body(body)
        return AgentSkill(
            metadata=_metadata_from_mapping(
                directory_name,
                content_hash,
                metadata,
            ),
            body=body.strip(),
        )


def _split_skill_text(text: str) -> tuple[str, str]:
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != "---":
        raise InvalidAgentSkillError("SKILL.md must start with frontmatter.")

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            if len(frontmatter) > MAX_SKILL_FRONTMATTER_CHARS:
                raise InvalidAgentSkillError("Skill frontmatter is too large.")
            return frontmatter, body
    raise InvalidAgentSkillError("SKILL.md frontmatter was not closed.")


def _parse_frontmatter(frontmatter: str) -> dict[str, object]:
    values: dict[str, object] = {}
    current_list_key: str | None = None
    current_mapping_key: str | None = None
    for raw_line in frontmatter.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("  - ") and current_list_key is not None:
            _append_list_value(values, current_list_key, raw_line[4:])
            continue
        if raw_line.startswith("  ") and current_mapping_key is not None:
            _append_mapping_value(values, current_mapping_key, raw_line[2:])
            continue
        current_list_key = None
        current_mapping_key = None
        key, value = _frontmatter_pair(raw_line)
        if value == "":
            if key in {"allowed-tools", "compatibility"}:
                values[key] = []
                current_list_key = key
            else:
                values[key] = {}
                current_mapping_key = key
        else:
            values[key] = _parse_frontmatter_value(value)
    return values


def _frontmatter_pair(line: str) -> tuple[str, str]:
    if line.startswith(" "):
        raise InvalidAgentSkillError("Unsupported frontmatter indentation.")
    key, separator, value = line.partition(":")
    if not separator or not key.strip():
        raise InvalidAgentSkillError("Invalid frontmatter key-value line.")
    return key.strip(), value.strip()


def _append_list_value(
    values: dict[str, object],
    key: str,
    raw_value: str,
) -> None:
    current = values.get(key)
    if not isinstance(current, list):
        raise InvalidAgentSkillError("Invalid frontmatter list.")
    items = cast("list[object]", current)
    items.append(_strip_quotes(raw_value.strip()))


def _append_mapping_value(
    values: dict[str, object],
    key: str,
    raw_line: str,
) -> None:
    current = values.get(key)
    if not isinstance(current, dict):
        raise InvalidAgentSkillError("Invalid frontmatter mapping.")
    child_key, child_value = _frontmatter_pair(raw_line)
    current[child_key] = _parse_frontmatter_value(child_value)


def _parse_frontmatter_value(value: str) -> object:
    if value.startswith("[") and value.endswith("]"):
        inner = value.removeprefix("[").removesuffix("]").strip()
        if not inner:
            return []
        return [
            _strip_quotes(item.strip())
            for item in inner.split(",")
            if item.strip()
        ]
    return _strip_quotes(value)


def _strip_quotes(value: str) -> str:
    if (
        len(value) >= MIN_QUOTED_STRING_LENGTH
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]
    return value


def _metadata_from_mapping(
    directory_name: str,
    content_hash: str,
    values: dict[str, object],
) -> AgentSkillMetadata:
    name = _required_text(values, "name")
    if name != directory_name:
        raise InvalidAgentSkillError(
            "Skill frontmatter name must match its directory name.",
        )

    metadata = values.get("metadata")
    version = _metadata_version(metadata)
    return AgentSkillMetadata(
        name=AgentSkillName(name),
        description=_required_text(values, "description"),
        content_hash=content_hash,
        version=version,
        allowed_tools=_text_tuple(values.get("allowed-tools")),
    )


def _metadata_version(metadata: object) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    values = cast("Mapping[object, object]", metadata)
    raw_version = values.get("version")
    if raw_version is None:
        return None
    version = str(raw_version).strip()
    return version or None


def _required_text(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidAgentSkillError(f"Skill frontmatter requires {key}.")
    return value.strip()


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidAgentSkillError("allowed-tools must be a string list.")
    values = cast("list[object]", value)
    return tuple(str(item).strip() for item in values if str(item).strip())


def _validate_body(body: str) -> None:
    if not body.strip():
        raise InvalidAgentSkillError("Skill instruction body is required.")
    if len(body) > MAX_SKILL_BODY_CHARS:
        raise InvalidAgentSkillError("Skill instruction body is too large.")
    if EXECUTABLE_FENCE_PATTERN.search(body):
        raise InvalidAgentSkillError(
            "Executable script blocks are disabled for Agent Skills.",
        )
