"""Tests for the workflow MCP stdio transport."""

import asyncio
from collections.abc import Callable
from typing import cast

import anyio
import mcp_types
import pytest
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from mcp.server import MCPServer
from mcp.shared.message import SessionMessage

from agent_team.infrastructure.mcp.server import stdio_transport


class _LowLevelServer:
    def __init__(self) -> None:
        self.run_called = False

    def create_initialization_options(self) -> object:
        return object()

    async def run(
        self,
        read_stream: object,
        write_stream: object,
        initialization_options: object,
    ) -> None:
        self.run_called = True
        assert read_stream is not None
        assert write_stream is not None
        assert initialization_options is not None
        await cast(
            "ObjectReceiveStream[SessionMessage | Exception]",
            read_stream,
        ).aclose()
        await cast(
            "ObjectSendStream[SessionMessage]",
            write_stream,
        ).aclose()


class _Server:
    def __init__(self) -> None:
        self._lowlevel_server = _LowLevelServer()

    @property
    def lowlevel_server(self) -> _LowLevelServer:
        return self._lowlevel_server


class TestStdioTransport:
    """Stdio transport behavior tests."""

    def test_run_mcp_server_stdio_delegates_to_lowlevel_server(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Run the public stdio adapter around the MCP low-level server."""
        server = _Server()

        async def stdin_reader(read_send: object) -> None:
            await cast(
                "ObjectSendStream[SessionMessage | Exception]",
                read_send,
            ).aclose()
            await asyncio.sleep(0)

        async def stdout_writer(write_receive: object) -> None:
            await cast(
                "ObjectReceiveStream[SessionMessage]",
                write_receive,
            ).aclose()
            await asyncio.sleep(0)

        monkeypatch.setattr(stdio_transport, "_stdin_reader", stdin_reader)
        monkeypatch.setattr(stdio_transport, "_stdout_writer", stdout_writer)

        asyncio.run(
            stdio_transport.run_mcp_server_stdio(cast("MCPServer", server)),
        )

        assert server.lowlevel_server.run_called is True

    def test_stdin_reader_sends_valid_session_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parse line-delimited JSON-RPC messages from stdin chunks."""
        chunks = [
            b'{"jsonrpc":"2.0","id":1,',
            b'"method":"ping"}\n',
            b"",
        ]

        async def read_chunk() -> bytes:
            return chunks.pop(0)

        monkeypatch.setattr(stdio_transport, "_read_stdin_chunk", read_chunk)

        async def run_reader() -> SessionMessage | Exception:
            send, receive = anyio.create_memory_object_stream[
                SessionMessage | Exception
            ](1)
            try:
                await stdio_transport._stdin_reader(  # pyright: ignore[reportPrivateUsage]
                    send,
                )
                return await receive.receive()
            finally:
                await receive.aclose()

        message = asyncio.run(run_reader())

        assert isinstance(message, SessionMessage)
        assert isinstance(message.message, mcp_types.JSONRPCRequest)
        assert message.message.method == "ping"

    def test_stdin_reader_sends_validation_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Send validation exceptions through the read stream."""
        chunks = [b"not-json\n", b""]

        async def read_chunk() -> bytes:
            return chunks.pop(0)

        monkeypatch.setattr(stdio_transport, "_read_stdin_chunk", read_chunk)

        async def run_reader() -> SessionMessage | Exception:
            send, receive = anyio.create_memory_object_stream[
                SessionMessage | Exception
            ](1)
            try:
                await stdio_transport._stdin_reader(  # pyright: ignore[reportPrivateUsage]
                    send,
                )
                return await receive.receive()
            finally:
                await receive.aclose()

        message = asyncio.run(run_reader())

        assert isinstance(message, Exception)

    def test_read_stdin_chunk_uses_event_loop_reader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Read available bytes from stdin file descriptor."""

        async def read_chunk() -> bytes:
            loop = asyncio.get_running_loop()

            def add_reader(
                _file_descriptor: int,
                callback: Callable[[], None],
            ) -> None:
                callback()

            def remove_reader(_file_descriptor: int) -> bool:
                return True

            def read(_file_descriptor: int, _size: int) -> bytes:
                return b"message\n"

            monkeypatch.setattr(loop, "add_reader", add_reader)
            monkeypatch.setattr(loop, "remove_reader", remove_reader)
            monkeypatch.setattr(stdio_transport.os, "read", read)
            return await stdio_transport._read_stdin_chunk()  # pyright: ignore[reportPrivateUsage]

        assert asyncio.run(read_chunk()) == b"message\n"

    def test_read_stdin_chunk_surfaces_read_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Surface operating-system read failures."""

        async def read_chunk() -> None:
            loop = asyncio.get_running_loop()

            def add_reader(
                _file_descriptor: int,
                callback: Callable[[], None],
            ) -> None:
                callback()

            def read(_file_descriptor: int, _size: int) -> bytes:
                raise OSError("read failed")

            def remove_reader(_file_descriptor: int) -> bool:
                return True

            monkeypatch.setattr(loop, "add_reader", add_reader)
            monkeypatch.setattr(loop, "remove_reader", remove_reader)
            monkeypatch.setattr(stdio_transport.os, "read", read)
            await stdio_transport._read_stdin_chunk()  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(OSError, match="read failed"):
            asyncio.run(read_chunk())

    def test_stdout_writer_writes_jsonrpc_lines(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Write session messages as line-delimited JSON-RPC."""
        writes: list[bytes] = []
        response = mcp_types.JSONRPCResponse(
            jsonrpc="2.0",
            id=1,
            result={},
        )

        def write(_file_descriptor: int, payload: bytes) -> int:
            writes.append(payload)
            return len(payload)

        monkeypatch.setattr(stdio_transport.os, "write", write)

        async def run_writer() -> None:
            send, receive = anyio.create_memory_object_stream[SessionMessage](
                1
            )
            async with send:
                await send.send(SessionMessage(response))
            await stdio_transport._stdout_writer(  # pyright: ignore[reportPrivateUsage]
                receive,
            )

        asyncio.run(run_writer())

        assert writes == [b'{"jsonrpc":"2.0","id":1,"result":{}}\n']
