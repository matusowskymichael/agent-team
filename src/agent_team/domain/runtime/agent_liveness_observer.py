"""Read-only runtime liveness observation boundary."""

from typing import Protocol

from agent_team.domain.runtime.agent_liveness_snapshot import (
    AgentLivenessSnapshot,
)


class AgentLivenessObserver(Protocol):
    """Observe sanitized runtime facts without controlling workflow state."""

    def observe(self, snapshot: AgentLivenessSnapshot) -> None:
        """Receive the latest immutable liveness metadata."""
        ...
