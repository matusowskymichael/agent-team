"""Local provider response watchdog failure."""


class AgentProviderTimeoutError(RuntimeError):
    """A local provider response exceeded its configured safety deadline."""
