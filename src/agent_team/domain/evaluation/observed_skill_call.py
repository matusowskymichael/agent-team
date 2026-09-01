"""Observed Agent Skill diagnostic call."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObservedSkillCall:
    """A model-visible Agent Skill load or resource read observed in a run."""

    tool_name: str
    skill_name: str
    status: str
    content_hash: str | None = None
    resource_name: str | None = None
