"""Activity must not hide stalled durable task advancement."""

import json
from dataclasses import replace

import pytest

from agent_team.application.runtime.agent_run_progress import AgentRunProgress
from agent_team.application.runtime.agent_task_snapshot import (
    AgentTaskSnapshot,
)
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.runtime.agent_lifecycle_phase import AgentLifecyclePhase
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.workflow.task_status import TaskStatus
from tests.unit.fakes.runtime.fake_monotonic_clock import FakeMonotonicClock


class TestAgentConvergence:
    """Stop activity churn and semantic failures without a repair cap."""

    def test_preexisting_workflow_state_cannot_reset_deadline(
        self,
        progress_verified_snapshot: AgentTaskSnapshot,
        fake_monotonic_clock: FakeMonotonicClock,
    ) -> None:
        """Resumed history is a baseline rather than new advancement."""
        progress = AgentRunProgress(4, clock=fake_monotonic_clock)
        progress.seed(progress_verified_snapshot)
        fake_monotonic_clock.advance(1801)

        progress.observe([], progress_verified_snapshot)

        with pytest.raises(AgentStalledError, match="deadline"):
            progress.require_progress()

    def test_terminal_phase_survives_observing_verification(
        self, progress_verified_snapshot: AgentTaskSnapshot
    ) -> None:
        """Completed state outranks the evidence that completed it."""
        progress = AgentRunProgress(4)
        snapshot = replace(
            progress_verified_snapshot,
            task=replace(
                progress_verified_snapshot.task, status=TaskStatus.COMPLETED
            ),
        )

        progress.observe([], snapshot)

        assert progress.convergence.phase is AgentLifecyclePhase.COMPLETED

    def test_strict_subcheck_improvement_refreshes_deadline(
        self,
        progress_verified_snapshot: AgentTaskSnapshot,
        fake_monotonic_clock: FakeMonotonicClock,
    ) -> None:
        """Passing one more required probe permits continued repair."""
        initial = progress_verified_snapshot
        assert initial.verification is not None
        evidence = initial.verification
        progress = AgentRunProgress(4, clock=fake_monotonic_clock)
        progress.seed(initial)
        fake_monotonic_clock.advance(1799)
        improved = replace(
            initial,
            verification=replace(
                evidence,
                id=evidence.id + 1,
                checks=(replace(evidence.checks[0], exit_code=0),),
            ),
        )

        progress.observe([], improved)
        fake_monotonic_clock.advance(2)
        progress.require_progress()

        assert progress.convergence.advancement_remaining() == 1798

    def test_repaired_required_check_improves_seeded_failure(
        self,
        progress_verified_snapshot: AgentTaskSnapshot,
        progress_invocation: ToolInvocationRecord,
        fake_monotonic_clock: FakeMonotonicClock,
    ) -> None:
        """A required check can improve prior failed verification evidence."""
        progress = AgentRunProgress(4, clock=fake_monotonic_clock)
        progress.seed(progress_verified_snapshot)
        fake_monotonic_clock.advance(1799)

        progress.observe(
            [
                replace(
                    progress_invocation,
                    tool_name="run_check",
                    result_preview=json.dumps(
                        {"name": "backend", "exit_code": 0, "timed_out": False}
                    ),
                )
            ],
            progress_verified_snapshot,
        )
        fake_monotonic_clock.advance(2)

        progress.require_progress()
        assert progress.convergence.advancement_remaining() == 1798

    def test_non_improving_failure_cannot_refresh_deadline(
        self,
        progress_verified_snapshot: AgentTaskSnapshot,
        fake_monotonic_clock: FakeMonotonicClock,
    ) -> None:
        """Fresh IDs, prose and timing never count as a better outcome."""
        initial = progress_verified_snapshot
        assert initial.verification is not None
        progress = AgentRunProgress(4, clock=fake_monotonic_clock)
        progress.seed(initial)
        fake_monotonic_clock.advance(1801)

        progress.observe(
            [],
            replace(
                initial,
                verification=replace(
                    initial.verification, id=22, feedback="New prose."
                ),
            ),
        )

        with pytest.raises(AgentStalledError, match="deadline"):
            progress.require_progress()

    def test_new_revision_needs_fresh_passing_subchecks(
        self,
        progress_invocation: ToolInvocationRecord,
        fake_monotonic_clock: FakeMonotonicClock,
    ) -> None:
        """A stale prior pass cannot establish improvement after a patch."""
        progress = AgentRunProgress(10, clock=fake_monotonic_clock)
        for index, name, exit_code in ((1, "first", 0), (2, "second", 1)):
            progress.observe(
                [
                    replace(
                        progress_invocation,
                        id=index,
                        tool_name="run_check",
                        result_preview=json.dumps(
                            {
                                "name": name,
                                "exit_code": exit_code,
                                "timed_out": False,
                            }
                        ),
                    )
                ],
                None,
            )
        fake_monotonic_clock.advance(100)
        progress.observe(
            [
                replace(
                    progress_invocation,
                    id=3,
                    tool_name="apply_patch",
                    classification=ToolClassification.MUTATING,
                    result_preview=json.dumps(
                        {
                            "path": "backend/auth.py",
                            "applied": True,
                            "before_hash": "before",
                            "after_hash": "after",
                        }
                    ),
                ),
                replace(
                    progress_invocation,
                    id=4,
                    tool_name="run_check",
                    result_preview=json.dumps(
                        {
                            "name": "second",
                            "exit_code": 0,
                            "timed_out": False,
                        }
                    ),
                ),
            ],
            None,
        )

        assert progress.convergence.advancement_remaining() == 1700

    def test_distinct_discovery_cannot_extend_convergence(
        self, progress_invocation: ToolInvocationRecord
    ) -> None:
        """Arbitrarily novel reads cannot repeatedly reset advancement."""
        progress = AgentRunProgress(4)
        for index in range(5):
            progress.observe(
                [
                    replace(
                        progress_invocation,
                        id=index,
                        arguments_hash=f"arguments-{index}",
                        result_hash=f"result-{index}",
                    )
                ],
                None,
            )
        with pytest.raises(AgentStalledError, match="advancement"):
            progress.require_progress()

    def test_patch_churn_is_activity(
        self, progress_invocation: ToolInvocationRecord
    ) -> None:
        """New revisions without checks do not establish convergence."""
        progress = AgentRunProgress(4)
        for index in range(5):
            progress.observe(
                [
                    replace(
                        progress_invocation,
                        id=index,
                        tool_name="apply_patch",
                        classification=ToolClassification.MUTATING,
                        result_preview=json.dumps(
                            {
                                "path": "backend/auth.py",
                                "applied": True,
                                "before_hash": f"before-{index}",
                                "after_hash": f"after-{index}",
                            }
                        ),
                    )
                ],
                None,
            )
        with pytest.raises(AgentStalledError, match="advancement"):
            progress.require_progress()

    @pytest.mark.parametrize("exit_codes", [(1, 1, 1), (1, 2, 1, 2, 1)])
    def test_equivalent_failures_across_revisions(
        self,
        progress_invocation: ToolInvocationRecord,
        exit_codes: tuple[int, ...],
    ) -> None:
        """Repeated equivalent failures and two-state cycles must stop."""
        progress = AgentRunProgress(10)
        for index, exit_code in enumerate(exit_codes):
            progress.observe(
                [
                    replace(
                        progress_invocation,
                        id=index * 2,
                        tool_name="apply_patch",
                        classification=ToolClassification.MUTATING,
                        result_preview=json.dumps(
                            {
                                "path": "backend/auth.py",
                                "applied": True,
                                "before_hash": f"before-{index}",
                                "after_hash": f"after-{index}",
                            }
                        ),
                    ),
                    replace(
                        progress_invocation,
                        id=index * 2 + 1,
                        tool_name="run_check",
                        result_preview=json.dumps(
                            {
                                "name": "backend",
                                "exit_code": exit_code,
                                "timed_out": False,
                            }
                        ),
                    ),
                ],
                None,
            )
        with pytest.raises(AgentStalledError, match="equivalent"):
            progress.require_progress()

    @pytest.mark.parametrize("exit_codes", [(1, 1, 1), (1, 2, 1, 2, 1)])
    def test_verification_failures_ignore_new_submission_and_patch_ids(
        self,
        progress_verified_snapshot: AgentTaskSnapshot,
        exit_codes: tuple[int, ...],
    ) -> None:
        """Changing submission identity cannot hide the same failing checks."""
        progress = AgentRunProgress(10)
        initial = progress_verified_snapshot
        assert initial.handoff is not None
        assert initial.verification is not None
        for index, exit_code in enumerate(exit_codes):
            snapshot = replace(
                initial,
                handoff=replace(initial.handoff, id=index + 1),
                verification=replace(
                    initial.verification,
                    id=index + 1,
                    submission_id=index + 1,
                    feedback=f"Attempt {index}; token=private",
                    checks=(
                        replace(
                            initial.verification.checks[0],
                            exit_code=exit_code,
                        ),
                    ),
                ),
            )
            progress.observe([], snapshot)
        with pytest.raises(AgentStalledError, match="equivalent"):
            progress.require_progress()
