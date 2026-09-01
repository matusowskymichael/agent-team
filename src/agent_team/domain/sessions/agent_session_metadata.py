"""Agent session metadata domain model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class AgentSessionMetadata:
    """Stored binding for a persistent local agent session."""

    session_id: str
    feature_id: int
    role: DevelopmentRole
    created_at: datetime
    updated_at: datetime
