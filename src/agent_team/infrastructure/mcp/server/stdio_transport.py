"""Stdio transport runner for the local workflow MCP server."""

import asyncio
import os

import anyio
import mcp_types
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from mcp.server import MCPServer
from mcp.shared.message import SessionMessage

STDIN_FILENO = 0
STDOUT_FILENO = 1
READ_CHUNK_SIZE = 4096


async def run_mcp_server_stdio(server: MCPServer) -> None:
    """Run an MCP server over line-delimited stdio."""
    read_send, read_stream = anyio.create_memory_object_stream[
        SessionMessage | Exception
    ](0)
    write_send, write_receive = anyio.create_memory_object_stream[
        SessionMessage
    ](0)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_stdin_reader, read_send)
        task_group.start_soon(_stdout_writer, write_receive)
        lowlevel_server = (
            server._lowlevel_server  # pyright: ignore[reportPrivateUsage]
        )
        await lowlevel_server.run(
            read_stream,
            write_send,
            lowlevel_server.create_initialization_options(),
        )
        task_group.cancel_scope.cancel()


async def _stdin_reader(
    read_send: ObjectSendStream[SessionMessage | Exception],
) -> None:
    buffer = ""
    async with read_send:
        while True:
            chunk = await _read_stdin_chunk()
            if chunk == b"":
                if buffer:
                    await _send_stdin_line(read_send, buffer)
                return

            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", maxsplit=1)
                await _send_stdin_line(read_send, f"{line}\n")


async def _send_stdin_line(
    read_send: ObjectSendStream[SessionMessage | Exception],
    line: str,
) -> None:
    try:
        message = mcp_types.jsonrpc_message_adapter.validate_json(
            line,
            by_name=False,
        )
    except Exception as error:
        await read_send.send(error)
        return
    await read_send.send(SessionMessage(message))


async def _read_stdin_chunk() -> bytes:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bytes] = loop.create_future()

    def on_readable() -> None:
        if future.done():
            return
        try:
            chunk = os.read(STDIN_FILENO, READ_CHUNK_SIZE)
        except OSError as error:
            future.set_exception(error)
            return
        future.set_result(chunk)

    loop.add_reader(STDIN_FILENO, on_readable)
    try:
        return await future
    finally:
        loop.remove_reader(STDIN_FILENO)


async def _stdout_writer(
    write_receive: ObjectReceiveStream[SessionMessage],
) -> None:
    async with write_receive:
        async for session_message in write_receive:
            payload = session_message.message.model_dump_json(
                by_alias=True,
                exclude_unset=True,
            )
            os.write(STDOUT_FILENO, f"{payload}\n".encode())
