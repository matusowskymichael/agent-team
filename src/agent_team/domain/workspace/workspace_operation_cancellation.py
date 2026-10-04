"""Cancellation boundary for owned workspace child operations."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkspaceOperationCancellation(Protocol):
    """Stop only active operations owned by a trusted workspace executor."""

    def cancel_pending_operations(self) -> None:
        """Terminate owned commands without cancelling unrelated services."""
        ...
