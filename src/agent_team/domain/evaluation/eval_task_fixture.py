"""Evaluation task fixture domain model."""

from dataclasses import dataclass

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.task_status import TaskStatus


@dataclass(frozen=True, slots=True)
class EvalTaskFixture:
    """Development task fixture for an isolated evaluation database."""

    feature_id: int
    title: str
    description: str
    assigned_role: DevelopmentRole
    status: TaskStatus
