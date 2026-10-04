"""Workflow MCP stdio transport with explicit subprocess ownership."""

import logging
import os
import signal
import sys
from collections.abc import AsyncGenerator
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    suppress,
)
from typing import override

import anyio
import mcp_types
from agents.mcp import MCPServerStdio
from agents.mcp.server import MCPStreamTransport
from anyio.abc import ObjectReceiveStream, ObjectSendStream, Process
from anyio.streams.text import TextReceiveStream
from mcp.client.stdio import get_default_environment
from mcp.shared.message import SessionMessage

PROCESS_REAP_SECONDS = 1.0
LOGGER = logging.getLogger(__name__)


class OwnedWorkflowMCPServer(MCPServerStdio):
    """Preserve SDK MCP behavior while owning the local process kill scope."""

    _process_group_id: int | None = None

    @override
    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[MCPStreamTransport]:
        """Open the existing MCP wire protocol with an owned subprocess."""
        return self._owned_streams()

    def force_cleanup(self) -> None:
        """Kill only the new process group created for this transport."""
        process_group_id = self._process_group_id
        if process_group_id is None:
            return
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
        self._process_group_id = None

    @asynccontextmanager
    async def _owned_streams(self) -> AsyncGenerator[MCPStreamTransport]:
        process = await anyio.open_process(
            [self.params.command, *self.params.args],
            env=get_default_environment() | (self.params.env or {}),
            cwd=self.params.cwd,
            stderr=sys.stderr,
            start_new_session=True,
        )
        self._process_group_id = process.pid
        read_send, read_receive = anyio.create_memory_object_stream[
            SessionMessage | Exception
        ](0)
        write_send, write_receive = anyio.create_memory_object_stream[
            SessionMessage
        ](0)
        try:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(
                    _read_messages,
                    process,
                    read_send,
                    self.params.encoding,
                    self.params.encoding_error_handler,
                )
                task_group.start_soon(
                    _write_messages,
                    process,
                    write_receive,
                    self.params.encoding,
                    self.params.encoding_error_handler,
                )
                try:
                    yield read_receive, write_send
                finally:
                    # Kill before an await: native task cancellation can
                    # interrupt even an AnyIO shield during SDK teardown.
                    self.force_cleanup()
                    task_group.cancel_scope.cancel()
                    with anyio.move_on_after(
                        PROCESS_REAP_SECONDS, shield=True
                    ) as cleanup:
                        await _close_process(process)
                    if cleanup.cancelled_caught:
                        LOGGER.warning("Owned MCP process reaping timed out.")
        finally:
            self.force_cleanup()
            read_send.close()
            read_receive.close()
            write_send.close()
            write_receive.close()


async def _close_process(process: Process) -> None:
    if process.stdin is not None:
        await process.stdin.aclose()
    if process.stdout is not None:
        await process.stdout.aclose()
    await process.wait()
    await process.aclose()


async def _read_messages(
    process: Process,
    sender: ObjectSendStream[SessionMessage | Exception],
    encoding: str,
    errors: str,
) -> None:
    if process.stdout is None:
        raise RuntimeError("Owned MCP process has no stdout pipe.")
    stream = TextReceiveStream(
        process.stdout, encoding=encoding, errors=errors
    )
    buffer = ""
    async with sender:
        try:
            async for chunk in stream:
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", maxsplit=1)
                    await sender.send(_decode_message(line))
        except anyio.BrokenResourceError, anyio.ClosedResourceError:
            return


def _decode_message(line: str) -> SessionMessage:
    try:
        message = mcp_types.jsonrpc_message_adapter.validate_json(
            line, by_name=False
        )
    except ValueError:
        raise ValueError("Owned workflow MCP message was malformed.") from None
    return SessionMessage(message)


async def _write_messages(
    process: Process,
    receiver: ObjectReceiveStream[SessionMessage],
    encoding: str,
    errors: str,
) -> None:
    if process.stdin is None:
        raise RuntimeError("Owned MCP process has no stdin pipe.")
    async with receiver:
        try:
            async for session_message in receiver:
                payload = session_message.message.model_dump_json(
                    by_alias=True, exclude_unset=True
                )
                await process.stdin.send(
                    f"{payload}\n".encode(encoding, errors=errors)
                )
        except anyio.BrokenResourceError, anyio.ClosedResourceError:
            return
