"""Task handoff model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class TaskHandoff:
    """Persisted structured developer handoff for verification."""

    id: int
    task_id: int
    agent_run_id: int
    submitted_by: DevelopmentRole
    attribution: str
    workspace_identity_hash: str | None
    implementation_summary: str
    changed_paths: tuple[str, ...]
    reused_symbols: tuple[str, ...]
    new_symbols: tuple[str, ...]
    reuse_notes: str
    checks_attempted: tuple[str, ...]
    limitations: str
    next_action: str
    created_at: datetime
