"""Run-local ownership of SDK function-tool cancellation cleanup."""

import asyncio
from copy import copy
from dataclasses import dataclass, field
from typing import Any, cast

from agents import FunctionTool, Tool
from agents.tool_context import ToolContext


@dataclass(slots=True)
class AgentToolTaskScope:
    """Drain owned callbacks that the SDK can detach after cancellation."""

    _tasks: set[asyncio.Task[object]] = field(
        default_factory=set[asyncio.Task[object]],
    )

    def wrap(self, tool: Tool) -> Tool:
        """Retain permissions and schemas while tracking invocation tasks."""
        if not isinstance(tool, FunctionTool):
            return tool
        delegate = tool.on_invoke_tool

        # Any matches the installed SDK's FunctionTool callback boundary.
        async def invoke(context: ToolContext[Any], arguments: str) -> object:
            current = asyncio.current_task()
            if current is not None:
                self._tasks.add(cast("asyncio.Task[object]", current))
            return await delegate(context, arguments)

        wrapped = copy(tool)
        wrapped.on_invoke_tool = invoke
        return wrapped

    async def close(self) -> None:
        """Cancel active callbacks once and await their owned child cleanup."""
        self.cancel()
        pending = {task for task in self._tasks if not task.done()}
        if pending:
            await asyncio.wait(pending)
        for task in self._tasks:
            if task.done() and not task.cancelled():
                task.exception()
        self._tasks.clear()
        # Allow the SDK parents waiting on these callbacks to finish too.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    def cancel(self) -> None:
        """Signal callbacks before waiting for unrelated resource cleanup."""
        for task in self._tasks:
            if not task.done() and task.cancelling() == 0:
                task.cancel()
