"""Bounded synchronous subprocess execution with explicit cancellation."""

import os
import signal
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CancellableCommandRunner:
    """Own and reap command process groups started by one adapter."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _processes: set[subprocess.Popen[str]] = field(
        default_factory=set[subprocess.Popen[str]],
    )
    _cancelled: threading.Event = field(default_factory=threading.Event)

    def run(
        self,
        command: tuple[str, ...],
        cwd: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Execute without shell expansion and reap after timeout or cancel."""
        with self._lock:
            if self._cancelled.is_set():
                raise InterruptedError("Workspace operation was cancelled.")
            process = subprocess.Popen(  # noqa: S603
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            self._processes.add(process)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout,
                stderr,
            )
        except subprocess.TimeoutExpired:
            _kill_owned_group(process)
            stdout, stderr = process.communicate(timeout=1.0)
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout,
                stderr=stderr,
            ) from None
        finally:
            with self._lock:
                self._processes.discard(process)

    def cancel_pending_operations(self) -> None:
        """Prevent new commands and kill only this runner's active groups."""
        with self._lock:
            self._cancelled.set()
            for process in self._processes:
                _kill_owned_group(process)


def _kill_owned_group(process: subprocess.Popen[str]) -> None:
    """Kill the session created for a command, including its descendants."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
