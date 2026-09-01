"""Tool invocation audit denial data."""

from dataclasses import dataclass

from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart


@dataclass(frozen=True, slots=True)
class ToolInvocationDenial:
    """Data required to record a denied tool invocation."""

    invocation: ToolInvocationStart
    error_type: str
    error_message: str
