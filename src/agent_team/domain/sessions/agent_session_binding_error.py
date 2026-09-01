"""Agent session binding domain error."""


class AgentSessionBindingError(ValueError):
    """Raised when a session is reused for another role or feature."""
