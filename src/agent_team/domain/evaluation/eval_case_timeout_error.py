"""Typed expiration of one evaluation candidate attempt."""


class EvalCaseTimeoutError(TimeoutError):
    """Candidate execution exceeded its configured safety deadline."""
