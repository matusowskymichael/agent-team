"""Shared application harness for agent execution."""

from asyncio import CancelledError
from dataclasses import dataclass, field, replace

from agent_team.application.audit.audit_sanitizer import (
    hash_text,
    sanitize_text,
)
from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.agent_run_progress import AgentRunProgress
from agent_team.application.runtime.agent_task_snapshot import (
    AgentTaskSnapshot,
)
from agent_team.application.sessions.agent_session_service import (
    AgentSessionService,
)
from agent_team.application.sessions.workspace_identity import (
    workspace_identity_hash,
)
from agent_team.application.skills.agent_skill_context_builder import (
    AgentSkillContextBuilder,
)
from agent_team.application.skills.agent_skill_service import (
    AgentSkillService,
)
from agent_team.application.workflow.task_verification_service import (
    TaskVerificationService,
)
from agent_team.domain.audit.agent_audit_repository import AgentAuditRepository
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_status import (
    ToolInvocationStatus,
)
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.context.agent_context_provider import (
    AgentContextProvider,
)
from agent_team.domain.runtime.agent_executor import AgentExecutor
from agent_team.domain.runtime.agent_implementation_status import (
    AgentImplementationStatus,
)
from agent_team.domain.runtime.agent_not_implemented_error import (
    AgentNotImplementedError,
)
from agent_team.domain.runtime.agent_output_blank_error import (
    AgentOutputBlankError,
)
from agent_team.domain.runtime.agent_output_incomplete_error import (
    AgentOutputIncompleteError,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_runtime import AgentRuntime
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.agent_turn_limit_error import (
    AgentTurnLimitError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.sessions.agent_session_metadata import (
    AgentSessionMetadata,
)
from agent_team.domain.workflow.development_task_not_found_error import (
    DevelopmentTaskNotFoundError,
)
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_submission_error import (
    TaskSubmissionError,
)
from agent_team.domain.workflow.task_verification_evidence import (
    TaskVerificationEvidence,
)
from agent_team.domain.workflow.workflow_repository import WorkflowRepository

BLANK_OUTPUT_RECOVERY_PROMPT = (
    "The previous final output was blank. Complete the original request now. "
    "Do not repeat any workflow mutation that has already succeeded. If "
    "information is missing, ask a concise question instead of returning "
    "blank output."
)


@dataclass(frozen=True, slots=True)
class AgentHarness(AgentExecutor):
    """Shared harness that applies role profiles before model execution."""

    runtime: AgentRuntime
    audit_repository: AgentAuditRepository
    session_service: AgentSessionService | None = None
    context_provider: AgentContextProvider | None = None
    skill_service: AgentSkillService | None = None
    task_verification_service: TaskVerificationService | None = None
    workflow_repository: WorkflowRepository | None = None
    skill_context_builder: AgentSkillContextBuilder = field(
        default_factory=AgentSkillContextBuilder,
    )
    profile_catalog: AgentProfileCatalog = field(
        default_factory=AgentProfileCatalog,
    )

    async def execute(self, task: AgentTask) -> AgentResult:
        """Run segments until useful output or verified task completion."""
        profile = self.profile_catalog.get_profile(task.role)
        if (
            profile.implementation_status
            is AgentImplementationStatus.PLACEHOLDER
        ):
            raise AgentNotImplementedError(
                f"The {profile.role.value} role is not implemented yet.",
            )
        if task.run_limits is not None:
            profile = replace(profile, run_limits=task.run_limits)
        session = self._prepare_session(task)
        progress = AgentRunProgress(
            profile.run_limits.max_no_progress_segments,
        )
        run = self._start_run(task, profile, session)
        try:
            result, reason = await self._execute_segments(
                task,
                profile,
                run,
                session,
                progress,
            )
            self._validate_output(result, run.id)
        except (CancelledError, KeyboardInterrupt) as error:
            self._fail_run(run.id, progress, error, "cancelled")
            raise
        except Exception as error:
            self._fail_run(run.id, progress, error, _termination_reason(error))
            raise
        self.audit_repository.record_run_progress(
            run.id,
            progress.segment_count,
            reason,
        )
        self.audit_repository.complete_run(
            run_id=run.id,
            output_hash=hash_text(result.response),
            output_excerpt=sanitize_text(result.response),
            generation_metadata=result.generation_metadata,
        )
        return result

    def _start_run(
        self,
        task: AgentTask,
        profile: AgentProfile,
        session: AgentSessionMetadata | None,
    ) -> AgentRunRecord:
        return self.audit_repository.start_run(
            AgentRunStart(
                role=profile.role,
                model=self.runtime.model_name,
                prompt_hash=hash_text(task.prompt),
                prompt_excerpt=sanitize_text(task.prompt),
                max_turns=profile.run_limits.segment_turns,
                total_turn_limit=profile.run_limits.max_turns,
                session_id=None if session is None else session.session_id,
                feature_id=task.feature_id,
                task_id=task.task_id,
                workspace_identity_hash=_workspace_identity_hash(task),
            ),
        )

    async def _execute_segments(
        self,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
        session: AgentSessionMetadata | None,
        progress: AgentRunProgress,
    ) -> tuple[AgentResult, str]:
        snapshot = self._task_snapshot(task)
        progress.seed(snapshot)
        skill_context = self._build_skill_context(profile)
        segment_task = task
        blank_recovered = False
        while True:
            segment_profile = _segment_profile(profile, progress.turns_used)
            progress.segment_count += 1
            self.audit_repository.record_run_progress(
                run.id,
                progress.segment_count,
            )
            result = await self.runtime.execute(
                segment_task,
                segment_profile,
                run,
                self._build_context(task, session),
                skill_context,
            )
            progress.turns_used += max(1, result.turns_used)
            verification = self._verify_pending_task(task)
            snapshot = self._task_snapshot(task)
            progress.observe(
                self.audit_repository.list_tool_invocations(run.id),
                snapshot,
            )
            terminal = (
                _terminal_result(result, snapshot)
                if progress.mutation_started or verification is not None
                else None
            )
            if terminal is not None:
                return terminal
            unfinished = _requires_completion(task, progress)
            if unfinished and (
                snapshot is None
                or (
                    snapshot.task.status is TaskStatus.VERIFICATION_PENDING
                    and self.task_verification_service is None
                )
            ):
                raise TaskSubmissionError(
                    "Developer continuation requires authoritative task "
                    "state and a configured deterministic verifier.",
                )
            if result.segment_exhausted or unfinished:
                progress.require_progress()
                segment_task = replace(
                    task,
                    continuation_context=progress.continuation_context(
                        snapshot,
                    ),
                )
                continue
            if (
                _is_blank_output(result)
                and not blank_recovered
                and self._blank_recovery_allowed(run.id)
            ):
                blank_recovered = True
                segment_task = replace(
                    _blank_recovery_task(task),
                    continuation_context=progress.continuation_context(
                        snapshot,
                    ),
                )
                continue
            if verification is not None:
                result = replace(
                    result,
                    response=_append_verification_summary(
                        result.response,
                        verification,
                    ),
                )
            return result, "completed"

    def _task_snapshot(self, task: AgentTask) -> AgentTaskSnapshot | None:
        if not _is_bound_developer(task):
            return None
        repository = self.workflow_repository
        if repository is None and self.task_verification_service is not None:
            repository = self.task_verification_service.repository
        if repository is None or task.task_id is None:
            return None
        current = repository.get_task(task.task_id)
        if current is None:
            raise DevelopmentTaskNotFoundError(
                f"Development task {task.task_id} was not found.",
            )
        if (
            current.feature_id != task.feature_id
            or current.assigned_role is not task.role
        ):
            raise TaskSubmissionError(
                "Task continuation does not match the trusted run binding.",
            )
        return AgentTaskSnapshot(
            current,
            repository.latest_task_handoff(current.id),
            repository.latest_task_verification(current.id),
        )

    def _validate_output(self, result: AgentResult, run_id: int) -> None:
        if _is_incomplete_output(result):
            if result.generation_metadata is not None:
                self.audit_repository.record_run_generation_metadata(
                    run_id=run_id,
                    output_hash=hash_text(result.response),
                    output_excerpt=sanitize_text(result.response),
                    generation_metadata=result.generation_metadata,
                )
            raise AgentOutputIncompleteError(
                "The model reached its output limit before completing the "
                "response.",
            )
        if _is_blank_output(result):
            raise AgentOutputBlankError("The model returned blank output.")

    def _fail_run(
        self,
        run_id: int,
        progress: AgentRunProgress,
        error: BaseException,
        reason: str,
    ) -> None:
        try:
            self.audit_repository.record_run_progress(
                run_id,
                progress.segment_count,
                reason,
            )
            self.audit_repository.fail_run(
                run_id=run_id,
                error_type=type(error).__name__,
                error_message=(
                    "Agent execution cancelled."
                    if reason == "cancelled"
                    else sanitize_text(error)
                ),
            )
        except Exception as finalization_error:
            raise finalization_error from error

    def _prepare_session(
        self,
        task: AgentTask,
    ) -> AgentSessionMetadata | None:
        if self.session_service is None:
            return None
        return self.session_service.prepare_session(
            feature_id=task.feature_id,
            role=task.role,
            requested_session_id=task.session_id,
            task_id=task.task_id,
            workspace_identity_hash=_workspace_identity_hash(task),
        )

    def _build_context(
        self,
        task: AgentTask,
        session: AgentSessionMetadata | None,
    ) -> AgentContextEnvelope | None:
        if task.feature_id is None or session is None:
            return None
        if self.context_provider is None:
            return None
        return self.context_provider.build_context(
            feature_id=task.feature_id,
            role=task.role,
            session_id=session.session_id,
            task_id=task.task_id,
            workspace_identity_hash=session.workspace_identity_hash,
        )

    def _build_skill_context(self, profile: AgentProfile) -> str | None:
        if self.skill_service is None:
            return None
        metadata = self.skill_service.list_available_metadata(profile)
        return self.skill_context_builder.build_context(metadata)

    def _blank_recovery_allowed(self, run_id: int) -> bool:
        invocations = self.audit_repository.list_tool_invocations(run_id)
        return not any(
            invocation.classification is ToolClassification.MUTATING
            and invocation.status is not ToolInvocationStatus.DENIED
            for invocation in invocations
        )

    def _verify_pending_task(
        self,
        task: AgentTask,
    ) -> TaskVerificationEvidence | None:
        if (
            self.task_verification_service is None
            or task.task_id is None
            or task.workspace_root is None
        ):
            return None
        return self.task_verification_service.verify_task_if_pending(
            task.task_id,
            task.workspace_root,
        )


def _is_incomplete_output(result: AgentResult) -> bool:
    metadata = result.generation_metadata
    return metadata is not None and metadata.objectively_truncated


def _is_blank_output(result: AgentResult) -> bool:
    return not result.response.strip()


def _blank_recovery_task(task: AgentTask) -> AgentTask:
    return replace(
        task,
        prompt=f"{task.prompt}\n\n{BLANK_OUTPUT_RECOVERY_PROMPT}",
    )


def _workspace_identity_hash(task: AgentTask) -> str | None:
    if task.workspace_root is None:
        return None
    return workspace_identity_hash(task.workspace_root)


def _append_verification_summary(
    response: str,
    evidence: TaskVerificationEvidence,
) -> str:
    check_names = ", ".join(check.name for check in evidence.checks) or "none"
    return (
        f"{response}\n\n"
        "Verification result: "
        f"{evidence.outcome.value} "
        f"({evidence.failure_classification.value}); "
        f"checks: {check_names}; "
        f"feedback: {evidence.feedback}"
    )


def _segment_profile(profile: AgentProfile, turns_used: int) -> AgentProfile:
    limits = profile.run_limits
    if limits.max_turns is None:
        return profile
    remaining = limits.max_turns - turns_used
    if remaining <= 0:
        raise AgentTurnLimitError(
            "The explicit diagnostic total-turn limit was reached.",
        )
    return replace(
        profile,
        run_limits=replace(
            limits,
            segment_turns=min(limits.segment_turns, remaining),
        ),
    )


def _is_bound_developer(task: AgentTask) -> bool:
    return (
        task.role
        in {
            DevelopmentRole.BACKEND_DEVELOPER,
            DevelopmentRole.FRONTEND_DEVELOPER,
        }
        and task.task_id is not None
        and task.feature_id is not None
        and task.workspace_root is not None
    )


def _requires_completion(task: AgentTask, progress: AgentRunProgress) -> bool:
    return _is_bound_developer(task) and progress.mutation_started


def _terminal_result(
    result: AgentResult,
    snapshot: AgentTaskSnapshot | None,
) -> tuple[AgentResult, str] | None:
    if snapshot is None or snapshot.task.status not in {
        TaskStatus.COMPLETED,
        TaskStatus.BLOCKED,
    }:
        return None
    status = snapshot.task.status.value
    response = (
        f"Task {snapshot.task.id} {status}."
        if result.segment_exhausted
        or _is_blank_output(result)
        or _is_incomplete_output(result)
        else result.response
    )
    if snapshot.verification is not None:
        response = _append_verification_summary(
            response, snapshot.verification
        )
    return replace(
        result,
        response=response,
        segment_exhausted=False,
        generation_metadata=(
            None
            if _is_incomplete_output(result)
            else result.generation_metadata
        ),
    ), f"task_{status}"


def _termination_reason(error: Exception) -> str:
    reasons: tuple[tuple[type[Exception], str], ...] = (
        (AgentStalledError, "stalled"),
        (AgentTurnLimitError, "turn_limit"),
        (AgentOutputBlankError, "blank_output"),
        (AgentOutputIncompleteError, "output_limit"),
        (TaskSubmissionError, "configuration_error"),
    )
    for error_type, reason in reasons:
        if isinstance(error, error_type):
            return reason
    if type(error).__name__ in {
        "OllamaUnavailableError",
        "OllamaModelUnavailableError",
        "OllamaModelCapabilityError",
    }:
        return "provider_error"
    return "runtime_error"
