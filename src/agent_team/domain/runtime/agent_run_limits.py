"""Agent run limit settings."""

from dataclasses import dataclass

MIN_NO_PROGRESS_SEGMENTS = 2


@dataclass(frozen=True, slots=True)
class AgentRunLimits:
    """Segment and stall settings with an optional total diagnostic limit."""

    max_turns: int | None = None
    segment_turns: int = 10
    max_no_progress_segments: int = 3

    def __post_init__(self) -> None:
        """Validate run limits."""
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        if self.segment_turns < 1:
            raise ValueError("segment_turns must be at least 1.")
        if self.max_no_progress_segments < MIN_NO_PROGRESS_SEGMENTS:
            raise ValueError("max_no_progress_segments must be at least 2.")
