"""Typed failure for an objectively stalled logical agent run."""


class AgentStalledError(RuntimeError):
    """Several consecutive segments produced no new authoritative progress."""
