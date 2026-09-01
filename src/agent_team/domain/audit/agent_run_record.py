"""Agent run audit record."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)
from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class AgentRunRecord:
    """Immutable audit record for one agent run."""

    id: int
    role: DevelopmentRole
    model: str
    status: AgentRunStatus
    prompt_hash: str
    prompt_excerpt: str
    started_at: datetime
    ended_at: datetime | None
    max_turns: int
    output_hash: str | None
    output_excerpt: str | None
    error_type: str | None
    error_message: str | None
    session_id: str | None
    feature_id: int | None
    generation_metadata: AgentGenerationMetadata | None = None
