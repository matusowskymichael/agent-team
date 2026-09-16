"""Typed failure for an explicitly configured diagnostic turn limit."""


class AgentTurnLimitError(RuntimeError):
    """A caller's optional total turn limit stopped the logical run."""
