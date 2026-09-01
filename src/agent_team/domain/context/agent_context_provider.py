"""Agent context provider port."""

from typing import Protocol

from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.development_role import DevelopmentRole


class AgentContextProvider(Protocol):
    """Port for building feature-scoped runtime context."""

    def build_context(
        self,
        feature_id: int,
        role: DevelopmentRole,
        session_id: str,
    ) -> AgentContextEnvelope:
        """Build authoritative context for one feature-scoped run."""
        ...
