"""Capability authorization error."""


class CapabilityDeniedError(RuntimeError):
    """Raised when a role is not allowed to use a capability."""
