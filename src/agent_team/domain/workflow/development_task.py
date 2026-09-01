"""Development task domain model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.task_status import TaskStatus


@dataclass(frozen=True, slots=True)
class DevelopmentTask:
    """A development task for a tracked feature."""

    id: int
    feature_id: int
    title: str
    description: str
    assigned_role: DevelopmentRole
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
