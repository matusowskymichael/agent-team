"""Agent session identity domain model."""

from dataclasses import dataclass

from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class AgentSessionId:
    """Validated persistent local session identity."""

    value: str


def derive_agent_session_id(role: DevelopmentRole, feature_id: int) -> str:
    """Derive a deterministic role-and-feature-scoped session ID."""
    return f"role-{role.value}-feature-{feature_id}"
