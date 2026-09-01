"""Fake agent audit repository for unit tests."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.audit.tool_invocation_denial import ToolInvocationDenial
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole


def _empty_run_records() -> dict[int, AgentRunRecord]:
    return {}


def _empty_tool_invocation_records() -> dict[int, ToolInvocationRecord]:
    return {}


@dataclass(slots=True)
class FakeAgentAuditRepository:
    """In-memory audit repository fake with injectable failures."""

    fail_start_run: bool = False
    fail_complete_run: bool = False
    fail_fail_run: bool = False
    fail_start_tool_invocation: bool = False
    fail_complete_tool_invocation: bool = False
    fail_fail_tool_invocation: bool = False
    fail_deny_tool_invocation: bool = False
    runs: dict[int, AgentRunRecord] = field(
        default_factory=_empty_run_records,
    )
    tool_invocations: dict[int, ToolInvocationRecord] = field(
        default_factory=_empty_tool_invocation_records,
    )
    start_run_calls: int = 0
    start_tool_invocation_calls: int = 0
    next_run_id: int = 1
    next_tool_invocation_id: int = 1

    def start_run(
        self,
        run: AgentRunStart,
    ) -> AgentRunRecord:
        """Create a started audit run record."""
        self.start_run_calls += 1
        if self.fail_start_run:
            raise RuntimeError("audit start failed")

        run_record = AgentRunRecord(
            id=self.next_run_id,
            role=run.role,
            model=run.model,
            status=AgentRunStatus.STARTED,
            prompt_hash=run.prompt_hash,
            prompt_excerpt=run.prompt_excerpt,
            started_at=datetime.now(UTC),
            ended_at=None,
            max_turns=run.max_turns,
            output_hash=None,
            output_excerpt=None,
            error_type=None,
            error_message=None,
            session_id=run.session_id,
            feature_id=run.feature_id,
            generation_metadata=None,
        )
        self.runs[run_record.id] = run_record
        self.next_run_id += 1
        return run_record

    def complete_run(
        self,
        run_id: int,
        output_hash: str,
        output_excerpt: str,
        generation_metadata: AgentGenerationMetadata | None = None,
    ) -> AgentRunRecord:
        """Finalize a run as completed."""
        if self.fail_complete_run:
            raise RuntimeError("run completion audit failed")
        run = replace(
            self.runs[run_id],
            status=AgentRunStatus.COMPLETED,
            ended_at=datetime.now(UTC),
            output_hash=output_hash,
            output_excerpt=output_excerpt,
            generation_metadata=generation_metadata,
        )
        self.runs[run_id] = run
        return run

    def fail_run(
        self,
        run_id: int,
        error_type: str,
        error_message: str,
    ) -> AgentRunRecord:
        """Finalize a run as failed."""
        if self.fail_fail_run:
            raise RuntimeError("run failure audit failed")
        run = replace(
            self.runs[run_id],
            status=AgentRunStatus.FAILED,
            ended_at=datetime.now(UTC),
            error_type=error_type,
            error_message=error_message,
        )
        self.runs[run_id] = run
        return run

    def record_run_generation_metadata(
        self,
        run_id: int,
        output_hash: str,
        output_excerpt: str,
        generation_metadata: AgentGenerationMetadata,
    ) -> AgentRunRecord:
        """Record generation metadata for a run."""
        run = replace(
            self.runs[run_id],
            output_hash=output_hash,
            output_excerpt=output_excerpt,
            generation_metadata=generation_metadata,
        )
        self.runs[run_id] = run
        return run

    def start_tool_invocation(
        self,
        invocation: ToolInvocationStart,
    ) -> ToolInvocationRecord:
        """Create an allowed tool invocation record."""
        self.start_tool_invocation_calls += 1
        if self.fail_start_tool_invocation:
            raise RuntimeError("tool audit start failed")
        return self._create_tool_invocation(
            invocation_start=invocation,
            status=ToolInvocationStatus.ALLOWED,
            error_type=None,
            error_message=None,
        )

    def complete_tool_invocation(
        self,
        invocation_id: int,
        result_hash: str,
        result_preview: str,
    ) -> ToolInvocationRecord:
        """Finalize a tool invocation as completed."""
        if self.fail_complete_tool_invocation:
            raise RuntimeError("tool completion audit failed")
        invocation = replace(
            self.tool_invocations[invocation_id],
            status=ToolInvocationStatus.COMPLETED,
            ended_at=datetime.now(UTC),
            result_hash=result_hash,
            result_preview=result_preview,
        )
        self.tool_invocations[invocation_id] = invocation
        return invocation

    def fail_tool_invocation(
        self,
        invocation_id: int,
        error_type: str,
        error_message: str,
    ) -> ToolInvocationRecord:
        """Finalize a tool invocation as failed."""
        if self.fail_fail_tool_invocation:
            raise RuntimeError("tool failure audit failed")
        invocation = replace(
            self.tool_invocations[invocation_id],
            status=ToolInvocationStatus.FAILED,
            ended_at=datetime.now(UTC),
            error_type=error_type,
            error_message=error_message,
        )
        self.tool_invocations[invocation_id] = invocation
        return invocation

    def deny_tool_invocation(
        self,
        denial: ToolInvocationDenial,
    ) -> ToolInvocationRecord:
        """Record a denied tool invocation."""
        if self.fail_deny_tool_invocation:
            raise RuntimeError("tool denial audit failed")
        return self._create_tool_invocation(
            invocation_start=denial.invocation,
            status=ToolInvocationStatus.DENIED,
            error_type=denial.error_type,
            error_message=denial.error_message,
        )

    def list_tool_invocations(
        self,
        run_id: int,
    ) -> list[ToolInvocationRecord]:
        """Return tool invocations for one agent run."""
        return [
            invocation
            for invocation in self.tool_invocations.values()
            if invocation.run_id == run_id
        ]

    def open_run(
        self,
        task: AgentTask | None = None,
        role: DevelopmentRole = DevelopmentRole.DELIVERY_MANAGER,
        feature_id: int | None = None,
    ) -> AgentRunRecord:
        """Create a started run for tool-invocation tests."""
        prompt = "Test prompt." if task is None else task.prompt
        return self.start_run(
            AgentRunStart(
                role=role,
                model="fake-model",
                prompt_hash="prompt-hash",
                prompt_excerpt=prompt,
                max_turns=6,
                session_id=None,
                feature_id=feature_id,
            ),
        )

    def _create_tool_invocation(
        self,
        invocation_start: ToolInvocationStart,
        status: ToolInvocationStatus,
        error_type: str | None,
        error_message: str | None,
    ) -> ToolInvocationRecord:
        invocation = ToolInvocationRecord(
            id=self.next_tool_invocation_id,
            run_id=invocation_start.run_id,
            server_name=invocation_start.server_name,
            tool_name=invocation_start.tool_name,
            classification=invocation_start.classification,
            status=status,
            arguments_hash=invocation_start.arguments_hash,
            arguments_preview_json=invocation_start.arguments_preview_json,
            result_hash=None,
            result_preview=None,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC)
            if status is ToolInvocationStatus.DENIED
            else None,
            error_type=error_type,
            error_message=error_message,
        )
        self.tool_invocations[invocation.id] = invocation
        self.next_tool_invocation_id += 1
        return invocation
