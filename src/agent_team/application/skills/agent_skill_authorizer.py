"""Application authorization for Agent Skill access."""

from dataclasses import dataclass

from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.skills.agent_skill_access_denied_error import (
    AgentSkillAccessDeniedError,
)
from agent_team.domain.skills.agent_skill_name import AgentSkillName


@dataclass(frozen=True, slots=True)
class AgentSkillAuthorizer:
    """Deny-by-default authorizer for role-scoped Agent Skills."""

    def authorize(
        self,
        profile: AgentProfile,
        skill_name: AgentSkillName,
    ) -> None:
        """Allow only skills assigned to the active immutable profile."""
        if skill_name not in profile.allowed_skill_names:
            raise AgentSkillAccessDeniedError(
                f"The {profile.role.value} role cannot access skill "
                f"{skill_name.value}.",
            )
