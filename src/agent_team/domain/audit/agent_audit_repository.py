"""Agent audit repository port."""

from typing import Protocol

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.tool_invocation_denial import ToolInvocationDenial
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)


class AgentAuditRepository(Protocol):
    """Persistence port for local agent audit records."""

    def start_run(
        self,
        run: AgentRunStart,
    ) -> AgentRunRecord:
        """Record the start of an agent run."""
        ...

    def complete_run(
        self,
        run_id: int,
        output_hash: str,
        output_excerpt: str,
        generation_metadata: AgentGenerationMetadata | None = None,
    ) -> AgentRunRecord:
        """Finalize an agent run as completed."""
        ...

    def fail_run(
        self,
        run_id: int,
        error_type: str,
        error_message: str,
    ) -> AgentRunRecord:
        """Finalize an agent run as failed."""
        ...

    def record_run_generation_metadata(
        self,
        run_id: int,
        output_hash: str,
        output_excerpt: str,
        generation_metadata: AgentGenerationMetadata,
    ) -> AgentRunRecord:
        """Record sanitized model-generation metadata for an agent run."""
        ...

    def start_tool_invocation(
        self,
        invocation: ToolInvocationStart,
    ) -> ToolInvocationRecord:
        """Record an allowed tool invocation before execution."""
        ...

    def complete_tool_invocation(
        self,
        invocation_id: int,
        result_hash: str,
        result_preview: str,
    ) -> ToolInvocationRecord:
        """Finalize a tool invocation as completed."""
        ...

    def fail_tool_invocation(
        self,
        invocation_id: int,
        error_type: str,
        error_message: str,
    ) -> ToolInvocationRecord:
        """Finalize a tool invocation as failed."""
        ...

    def deny_tool_invocation(
        self,
        denial: ToolInvocationDenial,
    ) -> ToolInvocationRecord:
        """Record a denied tool invocation without execution."""
        ...

    def list_tool_invocations(
        self,
        run_id: int,
    ) -> list[ToolInvocationRecord]:
        """Return tool invocations for one agent run."""
        ...
