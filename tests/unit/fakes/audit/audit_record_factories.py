"""Audit record factories for tests."""

from datetime import UTC, datetime

from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)
from agent_team.domain.runtime.development_role import DevelopmentRole


def make_agent_run_record(
    run_id: int = 1,
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
    generation_metadata: AgentGenerationMetadata | None = None,
) -> AgentRunRecord:
    """Build an agent run record for tests."""
    return AgentRunRecord(
        id=run_id,
        role=DevelopmentRole.DELIVERY_MANAGER,
        model="qwen3.5:9b",
        status=status,
        prompt_hash="prompt-hash",
        prompt_excerpt="Create a login feature.",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ended_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        max_turns=6,
        output_hash="output-hash",
        output_excerpt="Created feature 1.",
        error_type=None,
        error_message=None,
        session_id=None,
        feature_id=None,
        generation_metadata=generation_metadata,
    )


def make_tool_invocation_record(
    invocation_id: int = 1,
    run_id: int = 1,
) -> ToolInvocationRecord:
    """Build a tool invocation record for tests."""
    return ToolInvocationRecord(
        id=invocation_id,
        run_id=run_id,
        server_name="development_workflow",
        tool_name="create_feature",
        classification=ToolClassification.MUTATING,
        status=ToolInvocationStatus.COMPLETED,
        arguments_hash="arguments-hash",
        arguments_preview_json='{"title":"Login"}',
        result_hash="result-hash",
        result_preview='{"id":1}',
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ended_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        error_type=None,
        error_message=None,
    )
