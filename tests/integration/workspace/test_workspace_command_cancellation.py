"""Cancellation of owned workspace check processes."""

import asyncio
import os
from pathlib import Path

import pytest

from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)


class TestWorkspaceCommandCancellation:
    """Interrupt checks without leaving command subprocesses alive."""

    def test_cancellation_terminates_owned_check_process(
        self,
        tmp_path: Path,
    ) -> None:
        """Kill only a started trusted command and allow future checks."""
        marker = tmp_path / "started"
        executor = LocalWorkspaceExecutor(
            tmp_path,
            check_commands={
                "backend": (
                    "python",
                    "-c",
                    "import os,time,pathlib; "
                    "pathlib.Path('started').write_text(str(os.getpid())); "
                    "time.sleep(0.2)",
                ),
            },
        )

        async def exercise() -> int:
            worker = asyncio.create_task(
                asyncio.to_thread(
                    executor.run_check,
                    "backend",
                )
            )
            async with asyncio.timeout(2):
                while not marker.exists():
                    await asyncio.sleep(0.001)
                process_id = int(marker.read_text())
                executor.cancel_pending_operations()
                result = await worker
            assert result.exit_code != 0
            return process_id

        process_id = asyncio.run(exercise())
        with pytest.raises(ProcessLookupError):
            os.kill(process_id, 0)

    def test_command_timeout_terminates_owned_process(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The existing command deadline also reaps the owned process."""
        monkeypatch.setattr(
            "agent_team.infrastructure.workspace.local_workspace_executor."
            "CHECK_TIMEOUT_SECONDS",
            0.005,
        )
        executor = LocalWorkspaceExecutor(
            tmp_path,
            check_commands={
                "backend": ("python", "-c", "import time; time.sleep(60)"),
            },
        )
        result = executor.run_check("backend")
        assert result.timed_out is True
        assert result.exit_code == 124
