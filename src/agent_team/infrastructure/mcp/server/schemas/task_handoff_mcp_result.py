"""MCP result schema for a submitted task handoff."""

from typing import TypedDict

from agent_team.domain.runtime.development_role import DevelopmentRole


class TaskHandoffMcpResult(TypedDict):
    """Structured MCP representation of a persisted task handoff."""

    id: int
    task_id: int
    agent_run_id: int
    submitted_by: DevelopmentRole
    attribution: str
    implementation_summary: str
    changed_paths: list[str]
    reused_symbols: list[str]
    new_symbols: list[str]
    reuse_notes: str
    checks_attempted: list[str]
    limitations: str
    next_action: str
    created_at: str
