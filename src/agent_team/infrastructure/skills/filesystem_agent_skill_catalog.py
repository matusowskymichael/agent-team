"""Filesystem-backed Agent Skill catalog."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from agent_team.domain.skills.agent_skill import AgentSkill
from agent_team.domain.skills.agent_skill_metadata import AgentSkillMetadata
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)
from agent_team.infrastructure.skills.skill_markdown_parser import (
    SkillMarkdownParser,
)
from agent_team.infrastructure.skills.skill_path_policy import (
    SkillPathPolicy,
)

DEFAULT_SKILLS_ROOT = Path("skills")


@dataclass(frozen=True, slots=True)
class FilesystemAgentSkillCatalog:
    """Read local Agent Skills from the repository skills directory."""

    root: Path = DEFAULT_SKILLS_ROOT
    parser: SkillMarkdownParser = field(default_factory=SkillMarkdownParser)
    path_policy: SkillPathPolicy | None = None

    def list_metadata(self) -> tuple[AgentSkillMetadata, ...]:
        """Discover skill metadata without returning instruction bodies."""
        policy = self._policy()
        metadata: list[AgentSkillMetadata] = []
        seen: set[str] = set()
        for directory in policy.skill_directories():
            skill_file = policy.skill_file(directory)
            text = skill_file.read_text()
            item = self.parser.parse_metadata(
                directory_name=directory.name,
                content_hash=_hash_text(text),
                text=text,
            )
            if item.name.value in seen:
                raise InvalidAgentSkillError("Duplicate skill names fail.")
            seen.add(item.name.value)
            metadata.append(item)
        return tuple(metadata)

    def load_skill(self, name: AgentSkillName) -> AgentSkill:
        """Load one skill's instruction body."""
        policy = self._policy()
        directory = policy.skill_directory(name)
        skill_file = policy.skill_file(directory)
        text = skill_file.read_text()
        return self.parser.parse_skill(
            directory_name=directory.name,
            content_hash=_hash_text(text),
            text=text,
        )

    def read_resource(
        self,
        skill_name: AgentSkillName,
        relative_path: str,
    ) -> tuple[str, str]:
        """Read one validated file from a skill directory."""
        policy = self._policy()
        directory = policy.skill_directory(skill_name)
        resource = policy.resource_file(directory, relative_path)
        text = resource.read_text()
        return text, _hash_text(text)

    def _policy(self) -> SkillPathPolicy:
        return self.path_policy or SkillPathPolicy(self.root)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
