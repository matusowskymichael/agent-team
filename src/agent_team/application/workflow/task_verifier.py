"""Task verification application port."""

from pathlib import Path
from typing import Protocol

from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)


class TaskVerifier(Protocol):
    """Port for deterministic verification of a submitted task."""

    def verify(
        self,
        task: DevelopmentTask,
        handoff: TaskHandoff,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        """Run trusted verification for a task submission."""
        ...
