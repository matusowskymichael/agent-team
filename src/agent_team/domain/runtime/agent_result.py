"""Agent result domain model."""

from dataclasses import dataclass

from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """The final response produced by an agent."""

    response: str
    generation_metadata: AgentGenerationMetadata | None = None
