"""Focused continuation limits, advisory output, and failure regressions."""

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.agent_task_snapshot import (
    AgentTaskSnapshot,
)
from agent_team.application.sessions.workspace_identity import (
    workspace_identity_hash,
)
from agent_team.application.workflow.task_verification_service import (
    TaskVerificationService,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.agent_turn_limit_error import (
    AgentTurnLimitError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.development_task_not_found_error import (
    DevelopmentTaskNotFoundError,
)
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_submission_error import (
    TaskSubmissionError,
)
from agent_team.domain.workflow.task_verification_profiles import (
    default_verification_contract,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.runtime.fake_agent_runtime import FakeAgentRuntime
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


@dataclass
class _ClampedSegmentRuntime:
    """Record segment limits and discovery with optional final output."""

    audit: FakeAgentAuditRepository
    profiles: list[AgentProfile] = field(default_factory=list[AgentProfile])
    tasks: list[AgentTask] = field(default_factory=list[AgentTask])
    run_ids: list[int] = field(default_factory=list[int])
    model_name: str = "fake-model"
    results: tuple[AgentResult, ...] = ()

    async def execute(
        self,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
        context: AgentContextEnvelope | None = None,
        skill_context: str | None = None,
    ) -> AgentResult:
        """Record actual segment budgets and make distinct read progress."""
        assert context is None
        assert skill_context is None
        self.profiles.append(profile)
        self.tasks.append(task)
        self.run_ids.append(run.id)
        identifier = str(len(self.profiles))
        invocation = self.audit.start_tool_invocation(
            ToolInvocationStart(
                run_id=run.id,
                server_name="workspace",
                tool_name="read_file",
                classification=ToolClassification.READ_ONLY,
                arguments_hash=f"arguments-{identifier}",
                arguments_preview_json="{}",
            ),
        )
        self.audit.complete_tool_invocation(
            invocation.id,
            f"result-{identifier}",
            "{}",
        )
        if self.results:
            return self.results[len(self.profiles) - 1]
        return AgentResult(
            response="",
            segment_exhausted=True,
            turns_used=profile.run_limits.segment_turns,
        )


@dataclass
class _ExplodingVerifier:
    """Detect unintended verification after a failed model segment."""

    error: Exception
    calls: int = 0

    def verify(
        self,
        task: DevelopmentTask,
        handoff: TaskHandoff,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        """Raise an independent infrastructure error if invoked."""
        assert task.id == handoff.task_id
        assert workspace_root.is_absolute()
        self.calls += 1
        raise self.error


class TestAgentHarnessContinuation:
    """Keep diagnostic controls separate from ordinary logical completion."""

    @pytest.mark.parametrize("role", list(DevelopmentRole))
    def test_limits_default_to_unbounded_execution(
        self,
        role: DevelopmentRole,
    ) -> None:
        """Every profile inherits no total-turn or wall-clock deadline."""
        limits = AgentProfileCatalog().get_profile(role).run_limits

        assert limits.max_turns is None
        assert limits.segment_turns == 10
        assert limits.max_no_progress_segments == 3

    @pytest.mark.parametrize(
        ("field_name", "limit"),
        [
            ("max_turns", 0),
            ("max_turns", -1),
            ("segment_turns", 0),
            ("segment_turns", -1),
            ("max_no_progress_segments", 0),
            ("max_no_progress_segments", 1),
        ],
    )
    def test_invalid_limits_are_rejected(
        self,
        field_name: str,
        limit: int,
    ) -> None:
        """Reject nonpositive limits and a one-segment stall threshold."""
        with pytest.raises(ValueError, match=field_name):
            replace(AgentRunLimits(), **{field_name: limit})

    @pytest.mark.parametrize(
        ("limit", "segment_turns", "expected_segments"),
        [(7, 3, [3, 3, 1]), (2, 10, [2]), (6, 3, [3, 3])],
    )
    def test_total_limit_clamps_segments_and_audits_diagnostic_termination(
        self,
        limit: int,
        segment_turns: int,
        expected_segments: list[int],
    ) -> None:
        """Stop exactly at the explicit total cap despite useful progress."""
        audit = FakeAgentAuditRepository()
        runtime = _ClampedSegmentRuntime(audit)
        harness = AgentHarness(runtime=runtime, audit_repository=audit)
        task = AgentTask(
            prompt="Inspect relevant source files.",
            run_limits=AgentRunLimits(
                max_turns=limit,
                segment_turns=segment_turns,
            ),
        )

        with pytest.raises(AgentTurnLimitError, match="diagnostic"):
            asyncio.run(harness.execute(task))

        assert [
            profile.run_limits.segment_turns for profile in runtime.profiles
        ] == expected_segments
        assert all(
            received.prompt == task.prompt for received in runtime.tasks
        )
        assert runtime.tasks[0].continuation_context is None
        assert all(
            received.continuation_context is not None
            for received in runtime.tasks[1:]
        )
        assert runtime.run_ids == [1] * len(expected_segments)
        assert audit.start_run_calls == 1
        run = audit.runs[1]
        assert run.max_turns == segment_turns
        assert run.total_turn_limit == limit
        assert run.segment_count == len(expected_segments)
        assert run.status is AgentRunStatus.FAILED
        assert run.termination_reason == "turn_limit"
        assert run.error_type == "AgentTurnLimitError"

    @pytest.mark.parametrize(
        "task_status",
        [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.BLOCKED],
    )
    def test_bound_task_advice_preserves_model_answer(
        self,
        progress_snapshot: AgentTaskSnapshot,
        tmp_path: Path,
        task_status: TaskStatus,
    ) -> None:
        """Allow advisory answers when this run never begins a mutation."""
        current = replace(progress_snapshot.task, status=task_status)
        repository = FakeWorkflowRepository(tasks={current.id: current})
        runtime = FakeAgentRuntime(
            result=AgentResult(response="The task depends on the logout API."),
        )
        audit = FakeAgentAuditRepository()
        harness = AgentHarness(
            runtime=runtime,
            audit_repository=audit,
            workflow_repository=repository,
        )

        result = asyncio.run(
            harness.execute(
                AgentTask(
                    prompt="Explain task dependencies without editing.",
                    role=current.assigned_role,
                    feature_id=current.feature_id,
                    task_id=current.id,
                    workspace_root=tmp_path,
                )
            )
        )

        assert result == runtime.result
        assert runtime.execute_calls == 1
        assert repository.get_task(current.id) == current
        assert audit.list_tool_invocations(1) == []
        assert audit.runs[1].termination_reason == "completed"

    @pytest.mark.parametrize(
        "task_status",
        [TaskStatus.COMPLETED, TaskStatus.BLOCKED],
    )
    def test_terminal_advisory_continues_after_segment_exhaustion(
        self,
        progress_snapshot: AgentTaskSnapshot,
        tmp_path: Path,
        task_status: TaskStatus,
    ) -> None:
        """Wait for the advisory answer despite a preexisting terminal task."""
        current = replace(progress_snapshot.task, status=task_status)
        repository = FakeWorkflowRepository(tasks={current.id: current})
        audit = FakeAgentAuditRepository()
        expected = AgentResult(response="The logout API is the dependency.")
        runtime = _ClampedSegmentRuntime(
            audit,
            results=(
                AgentResult(
                    response="",
                    segment_exhausted=True,
                    turns_used=10,
                ),
                expected,
            ),
        )
        harness = AgentHarness(
            runtime=runtime,
            audit_repository=audit,
            workflow_repository=repository,
        )
        task = AgentTask(
            prompt="Explain task dependencies without editing.",
            role=current.assigned_role,
            feature_id=current.feature_id,
            task_id=current.id,
            workspace_root=tmp_path,
        )

        result = asyncio.run(harness.execute(task))

        assert result == expected
        assert runtime.run_ids == [1, 1]
        assert runtime.tasks[1].continuation_context is not None
        assert replace(runtime.tasks[1], continuation_context=None) == task
        assert repository.get_task(current.id) == current
        assert audit.runs[1].segment_count == 2
        assert audit.runs[1].status is AgentRunStatus.COMPLETED
        assert audit.runs[1].termination_reason == "completed"

    @pytest.mark.parametrize(
        ("mismatch", "error_type"),
        [
            ("feature", TaskSubmissionError),
            ("role", TaskSubmissionError),
            ("missing", DevelopmentTaskNotFoundError),
        ],
    )
    def test_rejects_invalid_bound_snapshot_before_model(
        self,
        progress_snapshot: AgentTaskSnapshot,
        tmp_path: Path,
        mismatch: str,
        error_type: type[Exception],
    ) -> None:
        """Do not expose task context outside the trusted feature or role."""
        current = progress_snapshot.task
        repository = FakeWorkflowRepository(tasks={current.id: current})
        runtime = FakeAgentRuntime(result=AgentResult("Unused."))
        audit = FakeAgentAuditRepository()
        task = AgentTask(
            prompt="Inspect assigned work.",
            role=current.assigned_role,
            feature_id=current.feature_id,
            task_id=current.id,
            workspace_root=tmp_path,
        )
        if mismatch == "feature":
            task = replace(task, feature_id=current.feature_id + 1)
        elif mismatch == "role":
            task = replace(task, role=DevelopmentRole.FRONTEND_DEVELOPER)
        else:
            task = replace(task, task_id=current.id + 1)
        harness = AgentHarness(
            runtime=runtime,
            audit_repository=audit,
            workflow_repository=repository,
        )

        with pytest.raises(error_type):
            asyncio.run(harness.execute(task))

        assert runtime.execute_calls == 0
        assert audit.runs[1].status is AgentRunStatus.FAILED
        assert audit.runs[1].segment_count == 0
        assert audit.runs[1].error_type == error_type.__name__

    @pytest.mark.parametrize("error_type", [RuntimeError, OSError])
    def test_provider_failure_is_not_masked_by_verifier_failure(
        self,
        progress_snapshot: AgentTaskSnapshot,
        tmp_path: Path,
        error_type: type[Exception],
    ) -> None:
        """Preserve the provider failure and leave its submission resumable."""
        current = replace(
            progress_snapshot.task,
            status=TaskStatus.IN_PROGRESS,
            verification_contract=default_verification_contract(
                progress_snapshot.task.assigned_role,
            ),
        )
        repository = FakeWorkflowRepository(tasks={current.id: current})
        handoff = repository.submit_task_handoff(
            TaskHandoffDraft(
                task_id=current.id,
                agent_run_id=1,
                submitted_by=current.assigned_role,
                attribution="trusted developer",
                workspace_identity_hash=workspace_identity_hash(tmp_path),
                implementation_summary="Implemented logout.",
                changed_paths=("src/auth.py",),
                reused_symbols=("AuthService.logout",),
                new_symbols=(),
                reuse_notes="Extended the existing method.",
                checks_attempted=("backend",),
                limitations="None.",
                next_action="Verify the submission.",
            ),
            TaskStatus.IN_PROGRESS,
            TaskStatus.VERIFICATION_PENDING,
        )
        assert handoff is not None
        verifier = _ExplodingVerifier(error_type("Verifier storage failed."))
        provider_error = OllamaUnavailableError("Local provider disconnected.")
        runtime = FakeAgentRuntime(
            result=AgentResult(response="unused"),
            error=provider_error,
        )
        audit = FakeAgentAuditRepository()
        harness = AgentHarness(
            runtime=runtime,
            audit_repository=audit,
            workflow_repository=repository,
            task_verification_service=TaskVerificationService(
                repository,
                verifier,
            ),
        )

        with pytest.raises(OllamaUnavailableError) as failure:
            asyncio.run(
                harness.execute(
                    AgentTask(
                        prompt="Finish the assigned task.",
                        role=current.assigned_role,
                        feature_id=current.feature_id,
                        task_id=current.id,
                        workspace_root=tmp_path,
                    )
                )
            )

        assert failure.value is provider_error
        assert runtime.execute_calls == 1
        assert verifier.calls == 0
        assert repository.latest_task_verification(current.id) is None
        assert repository.latest_task_handoff(current.id) == handoff
        stored = repository.get_task(current.id)
        assert stored is not None
        assert stored.status is TaskStatus.VERIFICATION_PENDING
        assert audit.runs[1].status is AgentRunStatus.FAILED
        assert audit.runs[1].termination_reason == "provider_error"
        assert audit.runs[1].error_type == "OllamaUnavailableError"

    def test_missing_verifier_fails_with_configuration_error(
        self,
        progress_snapshot: AgentTaskSnapshot,
        tmp_path: Path,
    ) -> None:
        """Do not accept a handoff final response without a verifier."""
        current = replace(
            progress_snapshot.task,
            status=TaskStatus.VERIFICATION_PENDING,
        )
        repository = FakeWorkflowRepository(tasks={current.id: current})
        runtime = FakeAgentRuntime(result=AgentResult(response="Submitted."))
        audit = FakeAgentAuditRepository()
        harness = AgentHarness(
            runtime=runtime,
            audit_repository=audit,
            workflow_repository=repository,
        )

        with pytest.raises(TaskSubmissionError, match="verif"):
            asyncio.run(
                harness.execute(
                    AgentTask(
                        prompt="Complete the assigned task.",
                        role=current.assigned_role,
                        feature_id=current.feature_id,
                        task_id=current.id,
                        workspace_root=tmp_path,
                    )
                )
            )

        assert runtime.execute_calls <= 1
        assert audit.runs[1].termination_reason == "configuration_error"
        assert repository.get_task(current.id) == current
