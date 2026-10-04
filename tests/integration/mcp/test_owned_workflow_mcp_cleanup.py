"""Owned MCP transport cleanup cannot outlive a short runtime grace."""

import asyncio
import os
import signal
import threading
from contextlib import suppress
from pathlib import Path

import pytest

from agent_team.infrastructure.mcp.client.authorized_mcp_server import (
    AuthorizedMCPServer,
)


class TestOwnedWorkflowMcpCleanup:
    """Terminate only owned subprocesses even when graceful exit fails."""

    @pytest.mark.parametrize(
        "stubborn_workflow_mcp", ["initializing"], indirect=True
    )
    def test_cancellation_during_initialization_reaps_owned_child(
        self,
        stubborn_workflow_mcp: tuple[AuthorizedMCPServer, Path],
    ) -> None:
        """Cancellation before initialization cannot leak a spawned server."""
        server, marker = stubborn_workflow_mcp
        server.force_cleanup()

        async def exercise() -> None:
            connection = asyncio.create_task(server.connect())
            async with asyncio.timeout(2):
                while not marker.exists():
                    await asyncio.sleep(0.001)
                connection.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await connection
            server.force_cleanup()
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(exercise())
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)

    @pytest.mark.parametrize(
        "stubborn_workflow_mcp", ["malformed"], indirect=True
    )
    def test_malformed_initialization_reaps_child_without_raw_payload(
        self,
        stubborn_workflow_mcp: tuple[AuthorizedMCPServer, Path],
    ) -> None:
        """Malformed stdout causes a bounded failure without leaking input."""
        server, marker = stubborn_workflow_mcp

        async def exercise() -> None:
            async with asyncio.timeout(2):
                with pytest.raises(Exception) as error:
                    await server.connect()
            assert "invalid-secret-json" not in str(error.value)
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(exercise())
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)

    def test_short_cleanup_grace_cannot_abandon_stubborn_owned_process(
        self,
        stubborn_workflow_mcp: tuple[AuthorizedMCPServer, Path],
    ) -> None:
        """A timeout before SDK escalation cannot strand its shielded pipes."""
        server, marker = stubborn_workflow_mcp
        fallback_used = threading.Event()
        finished = threading.Event()

        def stop_regression_hang() -> None:
            if not finished.wait(0.2):
                fallback_used.set()
                with suppress(ProcessLookupError):
                    os.killpg(int(marker.read_text()), signal.SIGKILL)

        async def exercise() -> None:
            await server.connect()
            tools = await server.list_tools()
            assert tools
            fallback = threading.Thread(target=stop_regression_hang)
            fallback.start()
            try:
                async with asyncio.timeout(0.01):
                    await server.cleanup()
            except TimeoutError:
                pass
            finally:
                finished.set()
                fallback.join(timeout=0.5)
            assert not fallback_used.is_set()
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(exercise())
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
