"""Tests for workflow MCP server adapter."""

import asyncio
from collections.abc import Mapping
from typing import cast

from mcp.types import CallToolResult

from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.infrastructure.mcp.server.workflow_mcp_server import (
    create_workflow_mcp_server,
)
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestWorkflowMcpServer:
    """Workflow MCP server behavior tests."""

    def test_lists_clear_workflow_tools(self) -> None:
        """Expose the expected workflow tools and schemas."""
        server = create_workflow_mcp_server(
            WorkflowService(repository=FakeWorkflowRepository()),
        )

        tools = asyncio.run(server.list_tools())
        tool_names = {tool.name for tool in tools}
        create_feature = next(
            tool for tool in tools if tool.name == "create_feature"
        )
        get_feature = next(
            tool for tool in tools if tool.name == "get_feature"
        )
        get_feature_overview = next(
            tool for tool in tools if tool.name == "get_feature_overview"
        )

        assert tool_names == {
            "create_feature",
            "get_feature",
            "get_feature_overview",
            "list_features",
            "add_artifact",
            "list_artifacts",
            "create_task",
            "list_tasks",
            "update_task_status",
        }
        assert create_feature.description is not None
        assert "status" in create_feature.input_schema["properties"]
        assert create_feature.output_schema is not None
        assert "created_at" in create_feature.output_schema["properties"]
        assert get_feature.description is not None
        assert "Metadata only" in get_feature.description
        assert get_feature_overview.description is not None
        assert "complete feature details" in get_feature_overview.description
        assert "artifacts and tasks" in get_feature_overview.description

    def test_calls_workflow_tools(self) -> None:
        """Translate MCP calls to workflow service operations."""
        server = create_workflow_mcp_server(
            WorkflowService(repository=FakeWorkflowRepository()),
        )

        feature_result = asyncio.run(
            server.call_tool(
                "create_feature",
                {
                    "title": "Build MCP server",
                    "description": "Store development work.",
                },
            ),
        )
        feature = _structured_content(feature_result)
        feature_id = feature["id"]
        assert isinstance(feature_id, int)
        assert "artifacts" not in feature
        assert "tasks" not in feature

        artifact_result = asyncio.run(
            server.call_tool(
                "add_artifact",
                {
                    "feature_id": feature_id,
                    "kind": "requirements",
                    "content": "Persist features and tasks.",
                    "created_by": "agent:business_analyst",
                },
            ),
        )
        artifact = _structured_content(artifact_result)

        overview_result = asyncio.run(
            server.call_tool(
                "get_feature_overview",
                {"feature_id": feature_id},
            ),
        )
        overview = _structured_content(overview_result)
        overview_artifacts = overview["artifacts"]
        overview_tasks = overview["tasks"]
        assert isinstance(overview_artifacts, list)
        assert isinstance(overview_tasks, list)
        assert overview_artifacts == [artifact]
        assert overview_tasks == []

        task_result = asyncio.run(
            server.call_tool(
                "create_task",
                {
                    "feature_id": feature_id,
                    "title": "Implement SQLite repository",
                    "description": "Create schema and mappings.",
                    "assigned_role": "backend_developer",
                },
            ),
        )
        task = _structured_content(task_result)
        task_id = task["id"]
        assert isinstance(task_id, int)

        updated_result = asyncio.run(
            server.call_tool(
                "update_task_status",
                {
                    "task_id": task_id,
                    "status": "completed",
                },
            ),
        )
        updated_task = _structured_content(updated_result)

        assert updated_task["status"] == "completed"


def _structured_content(result: object) -> Mapping[str, object]:
    assert isinstance(result, CallToolResult)
    assert result.is_error is False
    content = result.structured_content
    assert isinstance(content, Mapping)
    return cast("Mapping[str, object]", content)
