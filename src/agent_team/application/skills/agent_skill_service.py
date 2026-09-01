"""Application service for read-only Agent Skill access."""

from dataclasses import dataclass

from agent_team.application.skills.agent_skill_authorizer import (
    AgentSkillAuthorizer,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.skills.agent_skill import AgentSkill
from agent_team.domain.skills.agent_skill_catalog import AgentSkillCatalog
from agent_team.domain.skills.agent_skill_metadata import AgentSkillMetadata
from agent_team.domain.skills.agent_skill_name import AgentSkillName


@dataclass(frozen=True, slots=True)
class AgentSkillService:
    """Coordinate role-scoped Agent Skill discovery and loading."""

    catalog: AgentSkillCatalog
    authorizer: AgentSkillAuthorizer

    def list_available_metadata(
        self,
        profile: AgentProfile,
    ) -> tuple[AgentSkillMetadata, ...]:
        """Return only skill metadata assigned to the active profile."""
        allowed_names = profile.allowed_skill_names
        return tuple(
            metadata
            for metadata in self.catalog.list_metadata()
            if metadata.name in allowed_names
        )

    def load_skill(
        self,
        profile: AgentProfile,
        name: str,
    ) -> AgentSkill:
        """Load one assigned Agent Skill instruction body."""
        skill_name = self.authorize_skill_access(profile, name)
        return self.catalog.load_skill(skill_name)

    def read_skill_resource(
        self,
        profile: AgentProfile,
        skill_name: str,
        relative_path: str,
    ) -> tuple[str, str]:
        """Read one file contained within an assigned Agent Skill."""
        name = self.authorize_skill_access(profile, skill_name)
        return self.catalog.read_resource(name, relative_path)

    def authorize_skill_access(
        self,
        profile: AgentProfile,
        name: str,
    ) -> AgentSkillName:
        """Validate and authorize a skill name before file access."""
        skill_name = AgentSkillName(name)
        self.authorizer.authorize(profile, skill_name)
        return skill_name
