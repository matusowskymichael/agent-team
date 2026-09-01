"""Shared application harness for agent execution."""

from dataclasses import dataclass, field, replace

from agent_team.application.audit.audit_sanitizer import (
    hash_text,
    sanitize_error,
    sanitize_text,
)
from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.sessions.agent_session_service import (
    AgentSessionService,
)
from agent_team.application.skills.agent_skill_context_builder import (
    AgentSkillContextBuilder,
)
from agent_team.application.skills.agent_skill_service import (
    AgentSkillService,
)
from agent_team.domain.audit.agent_audit_repository import AgentAuditRepository
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
from agent_team.domain.runtime.agent_output_blank_error import (
    AgentOutputBlankError,
)
from agent_team.domain.runtime.agent_output_incomplete_error import (
    AgentOutputIncompleteError,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_runtime import AgentRuntime
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.sessions.agent_session_metadata import (
    AgentSessionMetadata,
)

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
    skill_context_builder: AgentSkillContextBuilder = field(
        default_factory=AgentSkillContextBuilder,
    )
    profile_catalog: AgentProfileCatalog = field(
        default_factory=AgentProfileCatalog,
    )

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute an agent task through a role-specific profile."""
        profile = self.profile_catalog.get_profile(task.role)
        session = self._prepare_session(task)
        context = self._build_context(task, session)
        skill_context = self._build_skill_context(profile)
        prompt_excerpt = sanitize_text(task.prompt)
        run = self.audit_repository.start_run(
            AgentRunStart(
                role=profile.role,
                model=self.runtime.model_name,
                prompt_hash=hash_text(task.prompt),
                prompt_excerpt=prompt_excerpt,
                max_turns=profile.run_limits.max_turns,
                session_id=None if session is None else session.session_id,
                feature_id=task.feature_id,
            ),
        )

        try:
            result = await self.runtime.execute(
                task,
                profile,
                run,
                context,
                skill_context,
            )
            if _is_blank_output(result) and self._blank_recovery_allowed(
                run.id,
            ):
                result = await self.runtime.execute(
                    _blank_recovery_task(task),
                    profile,
                    run,
                    context,
                    skill_context,
                )
        except Exception as error:
            error_type, error_message = sanitize_error(error)
            try:
                self.audit_repository.fail_run(
                    run_id=run.id,
                    error_type=error_type,
                    error_message=error_message,
                )
            except Exception as finalization_error:
                raise finalization_error from error
            raise

        output_excerpt = sanitize_text(result.response)
        if _is_incomplete_output(result):
            incomplete_error = AgentOutputIncompleteError(
                "The model reached its output limit before completing the "
                "response.",
            )
            error_type, error_message = sanitize_error(incomplete_error)
            if result.generation_metadata is not None:
                self.audit_repository.record_run_generation_metadata(
                    run_id=run.id,
                    output_hash=hash_text(result.response),
                    output_excerpt=output_excerpt,
                    generation_metadata=result.generation_metadata,
                )
            self.audit_repository.fail_run(
                run_id=run.id,
                error_type=error_type,
                error_message=error_message,
            )
            raise incomplete_error
        if _is_blank_output(result):
            blank_error = AgentOutputBlankError(
                "The model returned blank output.",
            )
            error_type, error_message = sanitize_error(blank_error)
            self.audit_repository.fail_run(
                run_id=run.id,
                error_type=error_type,
                error_message=error_message,
            )
            raise blank_error

        self.audit_repository.complete_run(
            run_id=run.id,
            output_hash=hash_text(result.response),
            output_excerpt=output_excerpt,
            generation_metadata=result.generation_metadata,
        )
        return result

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
