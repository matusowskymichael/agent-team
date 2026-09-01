"""Agent skill metadata domain record."""

from dataclasses import dataclass

from agent_team.domain.skills.agent_skill_name import AgentSkillName


@dataclass(frozen=True, slots=True)
class AgentSkillMetadata:
    """Portable Agent Skill metadata without instruction body content."""

    name: AgentSkillName
    description: str
    content_hash: str
    version: str | None = None
    allowed_tools: tuple[str, ...] = ()
