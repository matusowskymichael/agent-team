"""Candidate deadlines cancel and drain bounded asynchronous operations."""

import asyncio

import pytest

from agent_team.application.evaluation.async_eval_deadline import (
    AsyncEvalDeadline,
)
from agent_team.domain.evaluation.eval_case_timeout_error import (
    EvalCaseTimeoutError,
)


class TestAsyncEvalDeadline:
    """Retain explicit cancellation semantics and own candidate tasks."""

    def test_completed_candidate_returns_its_value(self) -> None:
        """Return a completed result without requesting cancellation."""

        async def operation() -> str:
            return "completed"

        assert (
            asyncio.run(AsyncEvalDeadline().run(operation(), 1, 1))
            == "completed"
        )

    def test_candidate_failure_remains_distinct_from_timeout(self) -> None:
        """Propagate candidate failures instead of relabeling them."""

        async def operation() -> str:
            raise ValueError("Expected candidate failure.")

        with pytest.raises(ValueError, match="Expected candidate"):
            asyncio.run(AsyncEvalDeadline().run(operation(), 1, 1))

    def test_timeout_drains_candidate_cleanup(self) -> None:
        """Complete cancellation cleanup before returning a timeout."""

        async def scenario() -> None:
            finished: list[bool] = []

            async def operation() -> str:
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0)
                    finished.append(True)
                return "unreachable"

            with pytest.raises(EvalCaseTimeoutError):
                await AsyncEvalDeadline().run(operation(), 0.001, 1)
            assert finished == [True]
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(scenario())

    @pytest.mark.parametrize("cleanup_raises", [False, True])
    def test_timeout_bounds_slow_cleanup_and_consumes_failures(
        self, cleanup_raises: bool
    ) -> None:
        """Keep timeout ownership when cleanup waits or raises an error."""

        async def scenario() -> None:
            finished: list[bool] = []

            async def operation() -> str:
                try:
                    await asyncio.Event().wait()
                finally:
                    try:
                        if cleanup_raises:
                            raise ValueError("Cleanup failed.")
                        await asyncio.Event().wait()
                    finally:
                        finished.append(True)
                return "unreachable"

            with pytest.raises(EvalCaseTimeoutError):
                await AsyncEvalDeadline().run(operation(), 0.001, 0.001)
            await asyncio.sleep(0)
            assert finished == [True]
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(scenario())

    @pytest.mark.parametrize("slow_cleanup", [False, True])
    def test_explicit_cancellation_never_becomes_a_timeout(
        self, slow_cleanup: bool
    ) -> None:
        """Preserve user cancellation even if cleanup exhausts its grace."""

        async def scenario() -> None:
            entered = asyncio.Event()
            finished: list[bool] = []

            async def operation() -> str:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    try:
                        if slow_cleanup:
                            await asyncio.Event().wait()
                    finally:
                        finished.append(True)
                return "unreachable"

            active = asyncio.create_task(
                AsyncEvalDeadline().run(operation(), 1, 0.001)
            )
            await entered.wait()
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            assert finished == [True]
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(scenario())
