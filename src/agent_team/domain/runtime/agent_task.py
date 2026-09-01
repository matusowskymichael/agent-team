"""Agent task domain model."""

from dataclasses import dataclass
from pathlib import Path

from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class AgentTask:
    """A user task to be completed by an agent."""

    prompt: str
    role: DevelopmentRole = DevelopmentRole.DELIVERY_MANAGER
    feature_id: int | None = None
    session_id: str | None = None
    task_id: int | None = None
    workspace_root: Path | None = None
