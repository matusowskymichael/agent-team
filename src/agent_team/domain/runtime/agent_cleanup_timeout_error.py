"""Failure to finish owned-resource cleanup within the safety grace."""


class AgentCleanupTimeoutError(RuntimeError):
    """Cleanup exceeded its deadline after owned termination was requested."""
