"""Deterministic model substitute with observable internal segment inputs."""

from dataclasses import dataclass, field

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask
from tests.integration.runtime.run_completion.completion_operations import (
    CompletionOperations,
)
from tests.integration.runtime.run_completion.completion_segment import (
    CompletionSegment,
)


@dataclass(slots=True)
class ScriptedCompletionRuntime:
    """Record trusted inputs and perform scripted operations exactly once."""

    operations: CompletionOperations
    segments: tuple[CompletionSegment, ...] = ()
    repeat_last: bool = False
    tasks: list[AgentTask] = field(default_factory=list[AgentTask])
    runs: list[AgentRunRecord] = field(default_factory=list[AgentRunRecord])
    contexts: list[AgentContextEnvelope | None] = field(
        default_factory=list[AgentContextEnvelope | None]
    )

    @property
    def model_name(self) -> str:
        """Expose a fake local model identity for persisted audit records."""
        return "fake-local-completion"

    async def execute(
        self,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
        context: AgentContextEnvelope | None = None,
        skill_context: str | None = None,
    ) -> AgentResult:
        """Execute the next segment without any SDK or model network calls."""
        assert skill_context is None
        index = len(self.tasks)
        self.tasks.append(task)
        self.runs.append(run)
        self.contexts.append(context)
        assert self.segments, "A deterministic script is required."
        assert index < len(self.segments) or self.repeat_last, (
            "Harness unexpectedly replayed a completed segment."
        )
        segment = self.segments[min(index, len(self.segments) - 1)]
        for operation in segment.operations:
            self.operations.perform(operation, task, profile, run)
        if segment.error is not None:
            raise segment.error
        return segment.result
