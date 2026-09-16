"""Progress, terminal workflow state and replay safety across segments."""

import asyncio
from dataclasses import replace

import pytest

from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.runtime.agent_turn_limit_error import (
    AgentTurnLimitError,
)
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from tests.integration.runtime.run_completion.completion_segment import (
    CompletionSegment,
)
from tests.integration.runtime.run_completion.run_completion_scenario import (
    RunCompletionScenario,
)

INCOMPLETE_TERMINAL_RESULTS = (
    AgentResult("Unfinished segment output.", segment_exhausted=True),
    AgentResult(" \n "),
    AgentResult(
        "Partial terminal result.",
        generation_metadata=AgentGenerationMetadata(
            finish_reason="length",
            input_tokens=100,
            output_tokens=100,
            visible_output_char_count=24,
            objectively_truncated=True,
            model="fake-local-completion",
        ),
    ),
)


class TestRunCompletion:
    """Keep one logical run alive while authoritative work advances."""

    def test_progress_across_internal_segments(
        self, run_completion_scenario: RunCompletionScenario
    ) -> None:
        """Allow forty unique discoveries without a total-turn ceiling."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            *(CompletionSegment(("discover",) * 10) for _ in range(4)),
            CompletionSegment(result=AgentResult("Discovery is complete.")),
        )

        result = asyncio.run(scenario.harness.execute(scenario.task))

        assert result.response == "Discovery is complete."
        assert len(scenario.runtime.tasks) == 5
        assert scenario.runtime.operations.discoveries == 40
        runs = scenario.audit.list_runs(limit=10)
        assert len(runs) == 1
        run = runs[0]
        assert run.status is AgentRunStatus.COMPLETED
        assert run.total_turn_limit is None
        assert run.segment_count == 5
        assert run.termination_reason is not None
        assert {item.id for item in scenario.runtime.runs} == {run.id}
        assert run.session_id is not None
        session = scenario.sessions.get_session(run.session_id)
        assert session is not None
        assert session.task_id == scenario.task.task_id
        assert session.workspace_identity_hash == run.workspace_identity_hash
        for task, context in zip(
            scenario.runtime.tasks, scenario.runtime.contexts, strict=True
        ):
            assert task.prompt == scenario.task.prompt
            assert task.role is scenario.task.role
            assert task.feature_id == scenario.task.feature_id
            assert task.task_id == scenario.task.task_id
            assert task.workspace_root == scenario.task.workspace_root
            assert task.session_id == scenario.task.session_id
            assert context is not None
            assert context.session_id == run.session_id
            assert (
                context.workspace_identity_hash == run.workspace_identity_hash
            )
        continuation = scenario.runtime.tasks[-1].continuation_context
        assert continuation is not None
        assert "omitted" in continuation
        assert "DISCOVERED_VALUE" not in continuation
        assert len(continuation) < 15_000

    @pytest.mark.parametrize(
        "segments",
        (
            (
                CompletionSegment(("activate", "patch")),
                CompletionSegment(
                    ("check", "submit"), AgentResult("Submitted.")
                ),
            ),
            (
                CompletionSegment(("activate", "patch", "check")),
                CompletionSegment(("submit",), AgentResult("Submitted.")),
            ),
            *(
                (
                    CompletionSegment(
                        ("activate", "patch", "check", "submit"), result
                    ),
                )
                for result in INCOMPLETE_TERMINAL_RESULTS
            ),
        ),
        ids=(
            "after-mutation",
            "before-submission",
            "exhausted-after-submission",
            "blank-after-submission",
            "truncated-after-submission",
        ),
    )
    def test_segment_exhaustion_preserves_mutations_and_handoff(
        self,
        run_completion_scenario: RunCompletionScenario,
        segments: tuple[CompletionSegment, ...],
    ) -> None:
        """Continue unfinished work and verify a submitted final segment."""
        scenario = run_completion_scenario
        scenario.runtime.segments = segments

        result = asyncio.run(scenario.harness.execute(scenario.task))

        assert result.response.strip()
        assert "Verification result: passed" in result.response
        assert not result.segment_exhausted
        assert result.generation_metadata is None
        assert len(scenario.runtime.tasks) == len(segments)
        assert scenario.runtime.operations.revision == 1
        assert len(scenario.verifier.submissions) == 1
        assert scenario.task.task_id is not None
        task = scenario.repository.get_task(scenario.task.task_id)
        assert task is not None
        assert task.status is TaskStatus.COMPLETED
        evidence = scenario.repository.latest_task_verification(task.id)
        assert evidence is not None
        assert evidence.outcome is TaskVerificationOutcome.PASSED
        run = scenario.audit.list_runs(limit=1)[0]
        assert run.status is AgentRunStatus.COMPLETED
        assert run.termination_reason == "task_completed"
        calls = scenario.audit.list_tool_invocations(run.id)
        assert [call.tool_name for call in calls] == [
            "update_task_status",
            "apply_patch",
            "run_check",
            "submit_task_for_verification",
        ]
        assert all(
            call.status is ToolInvocationStatus.COMPLETED for call in calls
        )

    def test_premature_final_after_mutation_continues_to_verification(
        self, run_completion_scenario: RunCompletionScenario
    ) -> None:
        """A model-written final cannot end a task still in progress."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            CompletionSegment(
                ("activate", "patch"), AgentResult("Implementation finished.")
            ),
            CompletionSegment(
                ("check", "submit"), AgentResult("Submitted for verification.")
            ),
        )

        result = asyncio.run(scenario.harness.execute(scenario.task))

        assert len(scenario.runtime.tasks) == 2
        assert "Verification result: passed" in result.response
        continuation = scenario.runtime.tasks[1].continuation_context
        assert continuation is not None
        assert "in_progress" in continuation
        assert "apply_patch" in continuation
        assert "backend/auth.py" in continuation
        context = scenario.runtime.contexts[1]
        assert context is not None
        assert "in_progress" in context.authoritative_context

    @pytest.mark.parametrize("segment_exhausted", (False, True))
    def test_repair_continues_beyond_two_failed_submissions(
        self,
        run_completion_scenario: RunCompletionScenario,
        segment_exhausted: bool,
    ) -> None:
        """Four failures permit a fifth meaningful repair in the same run."""
        scenario = run_completion_scenario
        scenario.verifier.required_revision = 5
        scenario.runtime.segments = (
            CompletionSegment(
                ("activate", "patch", "check", "submit"),
                AgentResult(
                    "Submitted revision 1.",
                    segment_exhausted=segment_exhausted,
                ),
            ),
            *(
                CompletionSegment(
                    ("patch", "check", "submit"),
                    AgentResult(
                        f"Submitted revision {revision}.",
                        segment_exhausted=segment_exhausted,
                    ),
                )
                for revision in range(2, 6)
            ),
        )

        result = asyncio.run(scenario.harness.execute(scenario.task))

        assert "Verification result: passed" in result.response
        assert len(scenario.runtime.tasks) == 5
        assert scenario.runtime.operations.revision == 5
        submissions = scenario.verifier.submissions
        assert len({handoff.id for handoff in submissions}) == 5
        assert len({handoff.agent_run_id for handoff in submissions}) == 1
        assert (
            len({handoff.workspace_identity_hash for handoff in submissions})
            == 1
        )
        assert len(scenario.audit.list_runs(limit=10)) == 1
        for task in scenario.runtime.tasks[1:]:
            continuation = task.continuation_context
            assert continuation is not None
            assert "Verification result: failed" in continuation
            assert "Repair to revision 5" in continuation
            assert "fixture-private-verification-value" not in continuation
            assert task.prompt == scenario.task.prompt
        assert scenario.task.task_id is not None
        task = scenario.repository.get_task(scenario.task.task_id)
        assert task is not None
        assert task.status is TaskStatus.COMPLETED
        evidence = scenario.repository.latest_task_verification(task.id)
        assert evidence is not None
        assert evidence.submission_id == submissions[-1].id

    def test_denied_patch_can_recover_after_activation(
        self, run_completion_scenario: RunCompletionScenario
    ) -> None:
        """Permit corrected authorized work after a denied premature patch."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            CompletionSegment(("denied_patch",)),
            CompletionSegment(
                ("activate", "patch", "check", "submit"),
                AgentResult("Submitted after activation."),
            ),
        )

        result = asyncio.run(scenario.harness.execute(scenario.task))

        assert "Verification result: passed" in result.response
        assert scenario.runtime.operations.revision == 1
        assert len(scenario.runtime.tasks) == 2
        run = scenario.audit.list_runs(limit=1)[0]
        calls = scenario.audit.list_tool_invocations(run.id)
        assert calls[0].status is ToolInvocationStatus.DENIED
        assert all(
            call.status is ToolInvocationStatus.COMPLETED for call in calls[1:]
        )

    @pytest.mark.parametrize(
        ("terminal_result", "expected_response"),
        (
            (AgentResult("Blocked on input."), "Blocked on input."),
            *(
                (result, "Task 1 blocked.")
                for result in INCOMPLETE_TERMINAL_RESULTS
            ),
        ),
        ids=("normal", "exhausted", "blank", "truncated"),
    )
    def test_valid_blocked_state_is_terminal(
        self,
        run_completion_scenario: RunCompletionScenario,
        terminal_result: AgentResult,
        expected_response: str,
    ) -> None:
        """Stop when an authorized task transition records a blocker."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            CompletionSegment(("activate", "patch")),
            CompletionSegment(("block",), terminal_result),
        )

        result = asyncio.run(scenario.harness.execute(scenario.task))

        assert result.response == expected_response
        assert not result.segment_exhausted
        assert result.generation_metadata is None
        assert scenario.task.task_id is not None
        task = scenario.repository.get_task(scenario.task.task_id)
        assert task is not None
        assert task.status is TaskStatus.BLOCKED
        assert len(scenario.runtime.tasks) == 2
        run = scenario.audit.list_runs(limit=1)[0]
        assert run.status is AgentRunStatus.COMPLETED
        assert run.termination_reason == "task_blocked"

    def test_diagnostic_total_limit_stays_explicit(
        self, run_completion_scenario: RunCompletionScenario
    ) -> None:
        """An opted-in limit remains distinct from no-progress detection."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (CompletionSegment(("discover",) * 10),)
        task = replace(scenario.task, run_limits=AgentRunLimits(max_turns=10))

        with pytest.raises(AgentTurnLimitError):
            asyncio.run(scenario.harness.execute(task))

        assert len(scenario.runtime.tasks) == 1
        run = scenario.audit.list_runs(limit=1)[0]
        assert run.error_type == "AgentTurnLimitError"


class TestRunCompletionStalls:
    """Preserve cancellation and provider failures while stopping loops."""

    @pytest.mark.parametrize("threshold", (2, 4))
    def test_identical_successful_reads_eventually_stall(
        self, run_completion_scenario: RunCompletionScenario, threshold: int
    ) -> None:
        """Repeated identical successful calls cannot refresh progress."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (CompletionSegment(("read_same",)),)
        scenario.runtime.repeat_last = True
        task = replace(
            scenario.task,
            run_limits=AgentRunLimits(max_no_progress_segments=threshold),
        )

        with pytest.raises(AgentStalledError):
            asyncio.run(scenario.harness.execute(task))

        assert len(scenario.runtime.tasks) == threshold + 1
        run = scenario.audit.list_runs(limit=1)[0]
        assert run.error_type == "AgentStalledError"
        assert run.status is AgentRunStatus.FAILED
        assert run.segment_count == threshold + 1
        assert "no new" in (run.error_message or "")
        assert scenario.runtime.operations.revision == 0

    def test_repeated_premature_denials_do_not_count_as_progress(
        self, run_completion_scenario: RunCompletionScenario
    ) -> None:
        """Keep activation enforcement and stop unchanged denied mutations."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (CompletionSegment(("denied_patch",)),)
        scenario.runtime.repeat_last = True

        with pytest.raises(AgentStalledError):
            asyncio.run(scenario.harness.execute(scenario.task))

        assert len(scenario.runtime.tasks) == 3
        assert scenario.runtime.operations.revision == 0
        assert scenario.task.task_id is not None
        task = scenario.repository.get_task(scenario.task.task_id)
        assert task is not None
        assert task.status is TaskStatus.PENDING
        run = scenario.audit.list_runs(limit=1)[0]
        calls = scenario.audit.list_tool_invocations(run.id)
        assert all(
            call.status is ToolInvocationStatus.DENIED for call in calls
        )

    def test_failed_patches_do_not_count_as_progress(
        self, run_completion_scenario: RunCompletionScenario
    ) -> None:
        """Applied-false results cannot prolong an unchanged active task."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            CompletionSegment(("activate",)),
            CompletionSegment(("failed_patch",)),
        )
        scenario.runtime.repeat_last = True

        with pytest.raises(AgentStalledError):
            asyncio.run(scenario.harness.execute(scenario.task))

        assert len(scenario.runtime.tasks) == 4
        assert scenario.runtime.operations.revision == 0
        assert not scenario.verifier.submissions

    @pytest.mark.parametrize(
        "error",
        (
            RuntimeError("Local provider failed."),
            asyncio.CancelledError(),
            KeyboardInterrupt(),
        ),
        ids=(
            "provider-failure",
            "explicit-cancellation",
            "keyboard-interrupt",
        ),
    )
    def test_failure_or_cancellation_after_mutation_never_replays(
        self,
        run_completion_scenario: RunCompletionScenario,
        error: BaseException,
    ) -> None:
        """Propagate separate failure causes without replaying a patch."""
        scenario = run_completion_scenario
        scenario.runtime.segments = (
            CompletionSegment(("activate", "patch"), error=error),
        )

        with pytest.raises(type(error)):
            asyncio.run(scenario.harness.execute(scenario.task))

        assert len(scenario.runtime.tasks) == 1
        assert scenario.runtime.operations.revision == 1
        assert scenario.task.task_id is not None
        task = scenario.repository.get_task(scenario.task.task_id)
        assert task is not None
        assert task.status is TaskStatus.IN_PROGRESS
        assert scenario.repository.latest_task_handoff(task.id) is None
        run = scenario.audit.list_runs(limit=1)[0]
        assert run.status is AgentRunStatus.FAILED
        assert run.error_type == type(error).__name__
