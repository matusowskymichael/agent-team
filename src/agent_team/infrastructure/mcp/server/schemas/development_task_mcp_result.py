"""MCP result schema for a development task."""

from typing import TypedDict

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.task_status import TaskStatus


class DevelopmentTaskMcpResult(TypedDict):
    """Structured MCP representation of a development task."""

    id: int
    feature_id: int
    title: str
    description: str
    assigned_role: DevelopmentRole
    status: TaskStatus
    created_at: str
    updated_at: str
