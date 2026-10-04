"""Aggregate check behavior with real manifests and isolated processes."""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from agent_team.infrastructure.workspace import (
    workspace_check_profile_runner as runner,
)


class TestWorkspaceCheckProfileRunner:
    """Select only trusted checks and stop immediately on check failure."""

    @pytest.mark.parametrize("arguments", ([], ["unknown"], ["backend", "x"]))
    def test_invalid_profile_is_rejected(
        self, arguments: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reject invalid profile selection without spawning a command."""
        assert runner.main(arguments) == 2
        assert "Usage:" in capsys.readouterr().err

    @pytest.mark.parametrize("failure_index", (None, 0, 2))
    def test_backend_runs_ordered_checks_and_stops_after_failure(
        self, monkeypatch: pytest.MonkeyPatch, failure_index: int | None
    ) -> None:
        """Keep aggregate checks deterministic and preserve failure codes."""
        calls: list[tuple[str, ...]] = []

        def run_command(
            command: Sequence[str], *, check: bool
        ) -> subprocess.CompletedProcess[bytes]:
            assert check is False
            calls.append(tuple(command))
            exit_code = 7 if len(calls) - 1 == failure_index else 0
            return subprocess.CompletedProcess(command, exit_code)

        monkeypatch.setattr(runner.subprocess, "run", run_command)
        monkeypatch.setattr(runner.sys, "argv", ["workspace-check", "backend"])
        assert runner.main() == (0 if failure_index is None else 7)
        expected = (
            runner.BACKEND_COMMANDS
            if failure_index is None
            else runner.BACKEND_COMMANDS[: failure_index + 1]
        )
        assert calls == list(expected)

    @pytest.mark.parametrize(
        ("manifest", "message"),
        (
            (None, "package.json was not found"),
            ("{", "not valid JSON"),
            ('{"scripts": []}', "no scripts object"),
            ('{"scripts": {"lint": "lint"}}', "missing script typecheck"),
        ),
    )
    def test_frontend_requires_complete_trusted_check_configuration(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        manifest: str | None,
        message: str,
    ) -> None:
        """Missing or malformed scripts never count as a passing check."""
        monkeypatch.chdir(tmp_path)
        if manifest is not None:
            (tmp_path / "package.json").write_text(manifest, encoding="utf-8")
        assert runner.main(["frontend"]) == 1
        errors = capsys.readouterr().err
        assert message in errors
        assert "No configured frontend checks" in errors

    def test_frontend_runs_only_named_allowlisted_scripts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ignore unrelated scripts while invoking the required suite."""
        scripts = dict.fromkeys((*runner.FRONTEND_SCRIPTS, "deploy"), "local")
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": scripts}), encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        calls: list[tuple[str, ...]] = []

        def run_command(
            command: Sequence[str], *, check: bool
        ) -> subprocess.CompletedProcess[bytes]:
            assert check is False
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(runner.subprocess, "run", run_command)
        assert runner.main(["frontend"]) == 0
        assert calls == [
            ("npm", "run", name) for name in runner.FRONTEND_SCRIPTS
        ]
