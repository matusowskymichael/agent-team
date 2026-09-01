"""Agent run audit start data."""

from dataclasses import dataclass

from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class AgentRunStart:
    """Data required to open an agent run audit record."""

    role: DevelopmentRole
    model: str
    prompt_hash: str
    prompt_excerpt: str
    max_turns: int
    session_id: str | None = None
    feature_id: int | None = None
