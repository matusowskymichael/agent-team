"""Loaded Agent Skill domain record."""

from dataclasses import dataclass

from agent_team.domain.skills.agent_skill_metadata import AgentSkillMetadata


@dataclass(frozen=True, slots=True)
class AgentSkill:
    """Loaded Agent Skill with procedural instruction body."""

    metadata: AgentSkillMetadata
    body: str
