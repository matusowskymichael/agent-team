"""Development task lookup error."""


class DevelopmentTaskNotFoundError(LookupError):
    """Raised when a requested development task does not exist."""
