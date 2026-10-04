"""Liveness snapshots contain bounded diagnostic facts rather than data."""

import json
from dataclasses import asdict, replace

import pytest

from agent_team.application.runtime.agent_run_progress import AgentRunProgress
from agent_team.application.runtime.agent_task_snapshot import (
    AgentTaskSnapshot,
)
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from tests.unit.fakes.runtime.fake_monotonic_clock import FakeMonotonicClock


class TestAgentLiveness:
    """Exclude source, hashes, secrets, absolute paths and arbitrary text."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        (
            ("backend/auth.py", ("backend/auth.py",)),
            ("/private/workspace/auth.py", ()),
            ("../private.py", ()),
            ("backend/token=private-value.py", ()),
            ("C:\\private\\auth.py", ()),
        ),
    )
    def test_changed_paths_use_safe_metadata_only(
        self,
        progress_invocation: ToolInvocationRecord,
        progress_verified_snapshot: AgentTaskSnapshot,
        path: str,
        expected: tuple[str, ...],
    ) -> None:
        """Expose safe relative paths without mutation contents or secrets."""
        progress = AgentRunProgress(4)
        progress.observe(
            [
                replace(
                    progress_invocation,
                    tool_name="apply_patch",
                    classification=ToolClassification.MUTATING,
                    result_preview=json.dumps(
                        {
                            "path": path,
                            "applied": True,
                            "before_hash": "private-before-hash",
                            "after_hash": "private-after-hash",
                            "new_text": "private-patch-content",
                        }
                    ),
                )
            ],
            progress_verified_snapshot,
        )

        snapshot = progress.liveness_snapshot(
            progress_verified_snapshot, "model"
        )
        rendered = json.dumps(asdict(snapshot))

        assert snapshot.changed_paths == expected
        assert snapshot.last_tool_name == "apply_patch"
        assert snapshot.last_verification_outcome == "failed"
        assert "private" not in rendered
        assert "hash" not in rendered
        assert "arguments" not in rendered
        assert "feedback" not in rendered
        assert "description" not in rendered

    def test_arbitrary_operation_and_waiting_text_are_not_exposed(
        self, progress_invocation: ToolInvocationRecord
    ) -> None:
        """Observer source allowlists tool names and waiting states."""
        progress = AgentRunProgress(4)
        progress.observe(
            [
                replace(
                    progress_invocation,
                    tool_name="private-operation token=fixture-secret",
                )
            ],
            None,
        )

        snapshot = progress.liveness_snapshot(
            None, "<think>private reasoning</think>"
        )

        assert snapshot.last_tool_name is None
        assert snapshot.waiting_phase == "model"
        assert "private" not in json.dumps(asdict(snapshot))

    def test_elapsed_liveness_uses_advancement_clock_without_mutation(
        self, fake_monotonic_clock: FakeMonotonicClock
    ) -> None:
        """Observability does not change convergence decisions or timers."""
        progress = AgentRunProgress(4, clock=fake_monotonic_clock)
        fake_monotonic_clock.advance(30)

        first = progress.liveness_snapshot(None, "model")
        fake_monotonic_clock.advance(30)
        second = progress.liveness_snapshot(None, "tool")

        assert first.elapsed_seconds == 30
        assert second.time_since_advancement_seconds == 60
        assert progress.convergence.advancement_count == 0
