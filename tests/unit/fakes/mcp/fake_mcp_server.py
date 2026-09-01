"""Fake MCP server for authorization tests."""

from typing import Any

from agents import AgentBase, RunContextWrapper
from agents.mcp import MCPServer
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, Tool

# Any is required by the installed Agents SDK MCPServer abstract methods.


class FakeMCPServer(MCPServer):
    """Fake MCP server that records delegated tool calls."""

    def __init__(
        self,
        tool_names: list[str],
        fail_call: bool = False,
    ) -> None:
        """Create a fake MCP server with named tools."""
        super().__init__(require_approval="never")
        self.tool_names = tool_names
        self.fail_call = fail_call
        self.call_count = 0
        self.received_calls: list[tuple[str, dict[str, Any] | None]] = []
        self.connected = False
        self.cleaned_up = False

    @property
    def name(self) -> str:
        """Return the fake server name."""
        return "fake_workflow"

    async def connect(self) -> None:
        """Mark the fake server connected."""
        self.connected = True

    async def cleanup(self) -> None:
        """Mark the fake server cleaned up."""
        self.cleaned_up = True

    async def list_tools(
        self,
        run_context: RunContextWrapper[Any] | None = None,
        agent: AgentBase | None = None,
    ) -> list[Tool]:
        """Return fake MCP tools."""
        _ = (run_context, agent)
        return [
            Tool(
                name=tool_name,
                input_schema=_input_schema(tool_name),
            )
            for tool_name in self.tool_names
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Record and return a fake tool result."""
        _ = meta
        self.call_count += 1
        self.received_calls.append((tool_name, arguments))
        if self.fail_call:
            raise RuntimeError("MCP tool failed.")
        return CallToolResult(content=[], is_error=False)

    async def list_prompts(self) -> ListPromptsResult:
        """Fail because prompts are not used by these tests."""
        raise NotImplementedError

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        """Fail because prompts are not used by these tests."""
        raise NotImplementedError


def _input_schema(tool_name: str) -> dict[str, Any]:
    if tool_name != "add_artifact":
        return {"type": "object", "properties": {}}
    return {
        "type": "object",
        "properties": {
            "feature_id": {"type": "integer"},
            "kind": {"type": "string"},
            "content": {"type": "string"},
            "created_by": {"type": "string"},
        },
        "required": ["feature_id", "kind", "content", "created_by"],
    }
