"""Incomplete agent output error."""


class AgentOutputIncompleteError(RuntimeError):
    """Raised when a model stops because its output limit was reached."""
