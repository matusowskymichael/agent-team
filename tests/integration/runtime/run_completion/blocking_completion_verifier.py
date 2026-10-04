"""Threaded verification fixture with explicit cancellation and drainage."""

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)


@dataclass(slots=True)
class BlockingCompletionVerifier:
    """Wait for the test to release cancellation cleanup in its worker."""

    started: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    before_wait: Callable[[], None] | None = None
    release_on_cancel: bool = False

    def cancel_pending_operations(self) -> None:
        """Signal cancellation independently of actual worker completion."""
        self.cancel_requested.set()
        if self.release_on_cancel:
            self.release.set()

    def verify(
        self,
        task: DevelopmentTask,
        handoff: TaskHandoff,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        """Represent a real synchronous verifier still using its workspace."""
        del task, handoff, workspace_root
        self.started.set()
        if self.before_wait is not None:
            self.before_wait()
        try:
            assert self.release.wait(timeout=1)
            raise asyncio.CancelledError
        finally:
            self.finished.set()
