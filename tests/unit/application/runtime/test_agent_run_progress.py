"""Objective progress, stall thresholds and bounded continuation context."""

import json
from dataclasses import replace

import pytest

from agent_team.application.runtime.agent_run_progress import AgentRunProgress
from agent_team.application.runtime.agent_task_snapshot import (
    AgentTaskSnapshot,
)
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.workflow.task_status import TaskStatus


class TestAgentRunProgress:
    """Detect repeated semantics without limiting legitimate work."""

    @pytest.mark.parametrize("threshold", [2, 3, 5])
    def test_stall_requires_configured_consecutive_empty_segments(
        self,
        threshold: int,
    ) -> None:
        """Ordinary exhaustion is not a stall before the full threshold."""
        progress = AgentRunProgress(threshold)
        for _ in range(threshold - 1):
            progress.observe([], None)
            progress.require_progress()
        progress.observe([], None)
        with pytest.raises(AgentStalledError, match="no new successful"):
            progress.require_progress()

    def test_unique_discovery_has_no_total_cap_and_resets_stall_window(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """Arbitrarily many new results each reset consecutive inactivity."""
        progress = AgentRunProgress(3)
        for index in range(50):
            progress.observe([], None)
            progress.observe([], None)
            progress.observe(
                [
                    replace(
                        progress_invocation, id=index, result_hash=str(index)
                    )
                ],
                None,
            )
            progress.require_progress()
            assert progress.no_progress_segments == 0

    def test_repeated_calls_and_results_ignore_new_audit_ids(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """New invocation IDs cannot make identical discovery into progress."""
        progress = AgentRunProgress(3)
        progress.observe([progress_invocation], None)
        for index in range(2, 5):
            progress.observe([replace(progress_invocation, id=index)], None)
        with pytest.raises(AgentStalledError):
            progress.require_progress()

    def test_allowed_invocation_can_later_complete_with_same_id(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """Do not permanently deduplicate an unfinished audit invocation."""
        progress = AgentRunProgress(3)
        progress.observe(
            [
                replace(
                    progress_invocation, status=ToolInvocationStatus.ALLOWED
                )
            ],
            None,
        )
        assert progress.no_progress_segments == 1
        progress.observe([progress_invocation], None)
        assert progress.no_progress_segments == 0

    @pytest.mark.parametrize(
        "status",
        [ToolInvocationStatus.DENIED, ToolInvocationStatus.FAILED],
    )
    def test_failed_or_denied_mutations_are_never_progress(
        self,
        progress_invocation: ToolInvocationRecord,
        status: ToolInvocationStatus,
    ) -> None:
        """Varying failed mutation arguments cannot prevent stall detection."""
        progress = AgentRunProgress(3)
        for index in range(3):
            progress.observe(
                [
                    replace(
                        progress_invocation,
                        id=index,
                        tool_name="apply_patch",
                        classification=ToolClassification.MUTATING,
                        status=status,
                        arguments_hash=str(index),
                    ),
                ],
                None,
            )
        with pytest.raises(AgentStalledError):
            progress.require_progress()
        assert "apply_patch succeeded" not in progress.continuation_context(
            None
        )

    @pytest.mark.parametrize(
        ("applied", "before", "after", "expected_progress"),
        [
            (True, "old", "new", True),
            (True, None, "new", True),
            (True, "same", "same", False),
            (False, "old", "new", False),
            (True, "old", None, False),
        ],
    )
    def test_only_confirmed_content_changes_count_as_patch_progress(
        self,
        progress_invocation: ToolInvocationRecord,
        applied: bool,
        before: str | None,
        after: str | None,
        expected_progress: bool,
    ) -> None:
        """Require a successful patch with a distinct resulting file state."""
        progress = AgentRunProgress(3)
        progress.observe(
            [
                replace(
                    progress_invocation,
                    tool_name="apply_patch",
                    classification=ToolClassification.MUTATING,
                    result_preview=json.dumps(
                        {
                            "path": "src/auth.py",
                            "applied": applied,
                            "before_hash": before,
                            "after_hash": after,
                        }
                    ),
                ),
            ],
            None,
        )
        assert (progress.no_progress_segments == 0) is expected_progress

    def test_patch_state_advances_with_same_arguments_and_new_after_hash(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """New file states progress while an already-seen state does not."""
        progress = AgentRunProgress(3)
        invocation = progress_invocation
        for index, after_hash in enumerate(["version-1", "version-2"]):
            invocation = replace(
                progress_invocation,
                id=index,
                tool_name="apply_patch",
                classification=ToolClassification.MUTATING,
                result_preview=json.dumps(
                    {
                        "path": "src/auth.py",
                        "applied": True,
                        "before_hash": "previous-version",
                        "after_hash": after_hash,
                    }
                ),
            )
            progress.observe([invocation], None)
            assert progress.no_progress_segments == 0
        progress.observe([replace(invocation, id=3)], None)
        assert progress.no_progress_segments == 1

    def test_handoff_prose_and_generated_ids_do_not_count_as_progress(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """Revised handoff prose cannot evade the no-progress threshold."""
        progress = AgentRunProgress(3)
        for index in range(4):
            invocation = replace(
                progress_invocation,
                id=index,
                tool_name="submit_task_for_verification",
                classification=ToolClassification.MUTATING,
                arguments_hash=f"prose-revision-{index}",
                arguments_preview_json=json.dumps(
                    {
                        "task_id": 4,
                        "changed_paths": ["src/auth.py"],
                        "checks_attempted": ["backend"],
                        "implementation_summary_hash": str(index),
                        "next_action_hash": str(index),
                    }
                ),
                result_hash=f"handoff-row-{index}",
            )
            progress.observe([invocation], None)
        with pytest.raises(AgentStalledError):
            progress.require_progress()

    def test_snapshot_seed_ignores_prior_work_and_activation_starts_mutation(
        self,
        progress_snapshot: AgentTaskSnapshot,
    ) -> None:
        """Only a newly observed status change counts after binding the run."""
        progress = AgentRunProgress(3)
        progress.seed(progress_snapshot)
        progress.observe([], progress_snapshot)
        assert progress.no_progress_segments == 1
        assert progress.mutation_started is False
        active = replace(
            progress_snapshot,
            task=replace(
                progress_snapshot.task, status=TaskStatus.IN_PROGRESS
            ),
        )
        progress.observe([], active)
        assert progress.no_progress_segments == 0
        assert progress.mutation_started is True

    def test_changed_paths_and_checks_survive_later_discovery_compaction(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """Keep mutation/check summaries separate from recent discovery."""
        progress = AgentRunProgress(3)
        progress.observe(
            [
                replace(
                    progress_invocation,
                    tool_name="apply_patch",
                    classification=ToolClassification.MUTATING,
                    result_preview=json.dumps(
                        {
                            "path": "src/auth.py",
                            "applied": True,
                            "before_hash": "old",
                            "after_hash": "new",
                        }
                    ),
                ),
                replace(
                    progress_invocation,
                    id=2,
                    tool_name="run_check",
                    result_preview=json.dumps(
                        {
                            "name": "backend",
                            "exit_code": 0,
                            "timed_out": False,
                            "stdout_excerpt": "private-check-output",
                        }
                    ),
                ),
            ],
            None,
        )
        for index in range(3, 33):
            progress.observe(
                [
                    replace(
                        progress_invocation, id=index, result_hash=str(index)
                    )
                ],
                None,
            )
        context = progress.continuation_context(None)
        assert "Successful changed paths: 1" in context
        assert "src/auth.py" in context
        assert "Completed trusted checks: 1" in context
        assert "backend; exit_code=0; timed_out=False" in context
        assert "16 older entries omitted" in context
        assert "private-source-content" not in context
        assert "private-arguments" not in context
        assert "private-check-output" not in context

    @pytest.mark.parametrize(
        ("status", "mutation_started"),
        [
            (TaskStatus.PENDING, False),
            (TaskStatus.IN_PROGRESS, False),
            (TaskStatus.VERIFICATION_PENDING, True),
            (TaskStatus.COMPLETED, False),
            (TaskStatus.BLOCKED, False),
        ],
    )
    def test_seed_recognizes_workflow_already_started(
        self,
        progress_snapshot: AgentTaskSnapshot,
        status: TaskStatus,
        mutation_started: bool,
    ) -> None:
        """Only a persisted pending handoff begins verification on seeding."""
        progress = AgentRunProgress(3)
        snapshot = replace(
            progress_snapshot,
            task=replace(progress_snapshot.task, status=status),
        )
        progress.seed(snapshot)
        progress.observe([], snapshot)
        assert progress.mutation_started is mutation_started
        assert progress.no_progress_segments == 1
        assert f"Current task 4 status: {status.value}" in (
            progress.continuation_context(snapshot)
        )

    def test_completed_check_progress_tracks_new_results_and_latest_status(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """A corrected check result advances while identical repeats do not."""
        progress = AgentRunProgress(3)
        for index, exit_code in enumerate([1, 0, 0]):
            progress.observe(
                [
                    replace(
                        progress_invocation,
                        id=index,
                        tool_name="run_check",
                        result_hash=f"exit-{exit_code}",
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
            assert progress.no_progress_segments == (1 if index == 2 else 0)
        context = progress.continuation_context(None)
        assert "Completed trusted checks: 1" in context
        assert "backend; exit_code=0; timed_out=False" in context
        assert "exit_code=1" not in context

    @pytest.mark.parametrize(
        "result",
        [
            None,
            "{invalid",
            "[]",
            "{}",
            '{"name":"backend","exit_code":true,"timed_out":false}',
            '{"name":"backend","exit_code":0,"timed_out":"false"}',
            '{"name":" ","exit_code":0,"timed_out":false}',
        ],
    )
    def test_missing_or_malformed_check_evidence_never_counts_as_progress(
        self,
        progress_invocation: ToolInvocationRecord,
        result: str | None,
    ) -> None:
        """Incomplete audit payloads cannot fabricate completed checks."""
        progress = AgentRunProgress(3)
        progress.seed(None)
        progress.observe(
            [
                replace(
                    progress_invocation,
                    tool_name="run_check",
                    result_preview=result,
                ),
            ],
            None,
        )
        assert progress.no_progress_segments == 1
        assert "Completed trusted checks: 0" in (
            progress.continuation_context(None)
        )

    def test_compaction_is_bounded_and_deterministic_for_many_operations(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """Bound every rendered collection while retaining all fingerprints."""
        invocations: list[ToolInvocationRecord] = []
        for index in range(40):
            invocations.extend(
                [
                    replace(
                        progress_invocation,
                        id=index * 2,
                        tool_name="apply_patch",
                        classification=ToolClassification.MUTATING,
                        result_preview=json.dumps(
                            {
                                "path": f"src/{index:03}.py",
                                "applied": True,
                                "before_hash": "old",
                                "after_hash": "new",
                            }
                        ),
                    ),
                    replace(
                        progress_invocation,
                        id=index * 2 + 1,
                        tool_name="run_check",
                        arguments_hash=str(index),
                        result_preview=json.dumps(
                            {
                                "name": f"check-{index:03}",
                                "exit_code": 0,
                                "timed_out": False,
                            }
                        ),
                    ),
                ]
            )
        first = AgentRunProgress(3)
        second = AgentRunProgress(3)
        first.observe(invocations, None)
        second.observe(list(reversed(invocations)), None)
        context = first.continuation_context(None)
        assert context == second.continuation_context(None)
        assert "Successful changed paths: 40 unique entries; 16 omitted" in (
            context
        )
        assert "Completed trusted checks: 40 unique entries; 28 omitted" in (
            context
        )
        assert "64 older entries omitted" in context
        assert len(context) < 6_000

    def test_volatile_check_output_cannot_reset_stall_window(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """Elapsed time and output hashes do not turn repeated checks novel."""
        progress = AgentRunProgress(3)
        for index in range(4):
            progress.observe(
                [
                    replace(
                        progress_invocation,
                        id=index,
                        tool_name="run_check",
                        result_hash=f"timing-output-{index}",
                        result_preview=json.dumps(
                            {
                                "name": "backend",
                                "exit_code": 0,
                                "timed_out": False,
                                "stdout_excerpt_hash": f"stdout-{index}",
                            }
                        ),
                    ),
                ],
                None,
            )
        with pytest.raises(AgentStalledError):
            progress.require_progress()
        assert "Completed trusted checks: 1" in (
            progress.continuation_context(None)
        )

    def test_new_patch_makes_check_of_changed_workspace_novel(
        self,
        progress_invocation: ToolInvocationRecord,
    ) -> None:
        """Checking changed code counts even when check output is stable."""
        progress = AgentRunProgress(3)
        check = replace(
            progress_invocation,
            tool_name="run_check",
            result_preview=json.dumps(
                {
                    "name": "backend",
                    "exit_code": 0,
                    "timed_out": False,
                }
            ),
        )
        patch = replace(
            progress_invocation,
            id=3,
            tool_name="apply_patch",
            classification=ToolClassification.MUTATING,
            result_preview=json.dumps(
                {
                    "path": "src/auth.py",
                    "applied": True,
                    "before_hash": "old",
                    "after_hash": "new",
                }
            ),
        )
        progress.observe([check], None)
        progress.observe([replace(check, id=2)], None)
        assert progress.no_progress_segments == 1
        progress.observe([patch], None)
        progress.observe([replace(check, id=4)], None)
        assert progress.no_progress_segments == 0
        progress.observe([replace(check, id=5)], None)
        assert progress.no_progress_segments == 1

    def test_volatile_verification_and_handoff_prose_do_not_reset_stalls(
        self,
        progress_verified_snapshot: AgentTaskSnapshot,
    ) -> None:
        """Persisted IDs, feedback and output churn are not semantic change."""
        progress = AgentRunProgress(3)
        progress.seed(progress_verified_snapshot)
        initial = progress_verified_snapshot
        assert initial.verification is not None
        assert initial.handoff is not None
        latest = initial
        for index in range(3):
            check = replace(
                initial.verification.checks[0],
                id=index + 2,
                stdout_hash=f"stdout-{index}",
                stderr_hash=f"stderr-{index}",
                stdout_excerpt=f"Tests ran in {index + 1} seconds.",
            )
            latest = replace(
                initial,
                handoff=replace(
                    initial.handoff,
                    id=index + 2,
                    implementation_summary=f"Attempt {index}.",
                ),
                verification=replace(
                    initial.verification,
                    id=index + 2,
                    submission_id=index + 2,
                    feedback=f"Attempt {index} has the same failing check.",
                    checks=(check,),
                ),
            )
            progress.observe([], latest)
        with pytest.raises(AgentStalledError):
            progress.require_progress()
        assert "Attempt 2 has the same failing check." in (
            progress.continuation_context(latest)
        )
        assert "Tests ran in" not in progress.continuation_context(latest)
