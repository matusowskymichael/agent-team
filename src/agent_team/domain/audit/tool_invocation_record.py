"""Tool invocation audit record."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus


@dataclass(frozen=True, slots=True)
class ToolInvocationRecord:
    """Immutable audit record for one MCP tool invocation."""

    id: int
    run_id: int
    server_name: str
    tool_name: str
    classification: ToolClassification
    status: ToolInvocationStatus
    arguments_hash: str
    arguments_preview_json: str
    result_hash: str | None
    result_preview: str | None
    started_at: datetime
    ended_at: datetime | None
    error_type: str | None
    error_message: str | None
