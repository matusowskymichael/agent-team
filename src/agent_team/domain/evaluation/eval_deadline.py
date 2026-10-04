"""Injectable asynchronous deadline boundary for candidate execution."""

from collections.abc import Awaitable
from typing import Protocol


class EvalDeadline(Protocol):
    """Bound execution and request cancellation before reporting timeout."""

    async def run[Result](
        self,
        operation: Awaitable[Result],
        timeout_seconds: float,
        cleanup_grace_seconds: float,
    ) -> Result:
        """Await one operation or cancel it within the cleanup grace."""
        ...
