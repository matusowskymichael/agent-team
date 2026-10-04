"""One deterministic internal execution segment for integration tests."""

from dataclasses import dataclass, field

from agent_team.domain.runtime.agent_result import AgentResult


@dataclass(frozen=True, slots=True)
class CompletionSegment:
    """Script operations and the boundary result without model inference."""

    operations: tuple[str, ...] = ()
    result: AgentResult = field(
        default_factory=lambda: AgentResult(
            response="", segment_exhausted=True, turns_used=10
        )
    )
    error: BaseException | None = None
