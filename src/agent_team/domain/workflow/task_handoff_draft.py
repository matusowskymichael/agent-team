"""Task handoff draft model."""

from dataclasses import dataclass

from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class TaskHandoffDraft:
    """Bounded developer handoff data before persistence."""

    task_id: int
    agent_run_id: int
    submitted_by: DevelopmentRole
    attribution: str
    workspace_identity_hash: str
    implementation_summary: str
    changed_paths: tuple[str, ...]
    reused_symbols: tuple[str, ...]
    new_symbols: tuple[str, ...]
    reuse_notes: str
    checks_attempted: tuple[str, ...]
    limitations: str
    next_action: str
