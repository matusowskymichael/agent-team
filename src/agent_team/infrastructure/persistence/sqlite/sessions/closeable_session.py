"""Closeable session protocol for Agents SDK sessions."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class CloseableSession(Protocol):
    """Protocol for SDK session implementations with close support."""

    def close(self) -> None:
        """Close the underlying session resources."""
