"""Agent skill name value object."""

import re
from dataclasses import dataclass

from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)

_VALID_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class AgentSkillName:
    """Validated portable Agent Skill name."""

    value: str

    def __post_init__(self) -> None:
        """Validate the Agent Skills directory-compatible name."""
        if not _VALID_SKILL_NAME.fullmatch(self.value):
            raise InvalidAgentSkillError(
                "Skill names must contain lowercase letters, digits, "
                "and hyphens only.",
            )

    def __str__(self) -> str:
        """Return the plain skill name."""
        return self.value
