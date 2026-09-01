"""Expected tool trajectory domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall


@dataclass(frozen=True, slots=True)
class ExpectedToolTrajectory:
    """One acceptable complete tool-call trajectory for a case."""

    required_tool_calls: tuple[ExpectedToolCall, ...]
    order_matters: bool = False
    optional_read_only_tool_calls: tuple[str, ...] = ()
    forbidden_tool_calls: tuple[str, ...] = ()
