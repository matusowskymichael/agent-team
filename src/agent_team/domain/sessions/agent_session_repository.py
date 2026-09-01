"""Agent session repository port."""

from typing import Protocol

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.sessions.agent_session_metadata import (
    AgentSessionMetadata,
)


class AgentSessionRepository(Protocol):
    """Persistence port for local feature-scoped session bindings."""

    def get_session(
        self,
        session_id: str,
    ) -> AgentSessionMetadata | None:
        """Return stored session metadata, if it exists."""
        ...

    def create_session(
        self,
        session_id: str,
        feature_id: int,
        role: DevelopmentRole,
    ) -> AgentSessionMetadata:
        """Persist a new local session binding."""
        ...

    def touch_session(self, session_id: str) -> AgentSessionMetadata:
        """Update a session timestamp and return its metadata."""
        ...
