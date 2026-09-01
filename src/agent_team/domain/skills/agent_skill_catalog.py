"""Agent Skill catalog port."""

from typing import Protocol

from agent_team.domain.skills.agent_skill import AgentSkill
from agent_team.domain.skills.agent_skill_metadata import AgentSkillMetadata
from agent_team.domain.skills.agent_skill_name import AgentSkillName


class AgentSkillCatalog(Protocol):
    """Read-only port for discovering and loading local Agent Skills."""

    def list_metadata(self) -> tuple[AgentSkillMetadata, ...]:
        """Return metadata for every valid local skill."""
        ...

    def load_skill(self, name: AgentSkillName) -> AgentSkill:
        """Return the full instruction body for one skill."""
        ...

    def read_resource(
        self,
        skill_name: AgentSkillName,
        relative_path: str,
    ) -> tuple[str, str]:
        """Return resource content and SHA-256 hash for one skill file."""
        ...
