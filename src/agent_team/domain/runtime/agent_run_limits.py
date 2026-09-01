"""Agent run limit settings."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentRunLimits:
    """Immutable limits for one agent run."""

    max_turns: int = 6

    def __post_init__(self) -> None:
        """Validate run limits."""
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
