"""Internal execution segment watchdog failure."""


class AgentSegmentTimeoutError(RuntimeError):
    """A segment timed out with an uncertain, safely resumable outcome."""
