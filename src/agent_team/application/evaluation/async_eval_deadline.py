"""Async candidate deadlines with bounded cancellation cleanup."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass

from agent_team.domain.evaluation.eval_case_timeout_error import (
    EvalCaseTimeoutError,
)


@dataclass(frozen=True, slots=True)
class AsyncEvalDeadline:
    """Apply a candidate deadline without retrying uncertain operations."""

    async def run[Result](
        self,
        operation: Awaitable[Result],
        timeout_seconds: float,
        cleanup_grace_seconds: float,
    ) -> Result:
        """Cancel timed-out candidates and wait only for the cleanup grace."""
        candidate = asyncio.ensure_future(operation)
        try:
            done, _ = await asyncio.wait(
                (candidate,),
                timeout=timeout_seconds,
            )
            if candidate in done:
                return candidate.result()
            cleaned_up = await _cancel_candidate(
                candidate, cleanup_grace_seconds
            )
            raise EvalCaseTimeoutError(
                "The evaluation candidate-case safety deadline was reached. "
                "Recorded tool outcomes are preserved; uncertain mutations "
                "must be reconciled before resuming."
                + (
                    ""
                    if cleaned_up
                    else " Candidate cleanup exhausted its grace period."
                ),
            )
        except asyncio.CancelledError, KeyboardInterrupt:
            await _cancel_candidate(candidate, cleanup_grace_seconds)
            raise


async def _cancel_candidate[Result](
    candidate: asyncio.Future[Result],
    grace_seconds: float,
) -> bool:
    """Request immediate cancellation and bound cooperative finalization."""
    if candidate.done():
        return True
    candidate.cancel()
    done, _ = await asyncio.wait((candidate,), timeout=grace_seconds)
    if candidate not in done:
        candidate.cancel()
        candidate.add_done_callback(_consume_candidate_error)
        await asyncio.sleep(0)
        return False
    if not candidate.cancelled():
        candidate.exception()
    return True


def _consume_candidate_error[Result](
    candidate: asyncio.Future[Result],
) -> None:
    """Retrieve a late cleanup error without changing terminal status."""
    if not candidate.cancelled():
        candidate.exception()
