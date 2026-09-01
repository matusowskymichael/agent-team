"""Tool invocation audit start data."""

from dataclasses import dataclass

from agent_team.domain.audit.tool_classification import ToolClassification


@dataclass(frozen=True, slots=True)
class ToolInvocationStart:
    """Data required to open a tool invocation audit record."""

    run_id: int
    server_name: str
    tool_name: str
    classification: ToolClassification
    arguments_hash: str
    arguments_preview_json: str
