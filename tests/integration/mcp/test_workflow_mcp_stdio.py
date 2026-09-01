"""Integration tests for the workflow MCP stdio server."""

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from mcp import Client, StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_process_options as mcp_options,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_config as workflow_mcp_config,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_factory as workflow_mcp_factory,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)


class TestWorkflowMcpStdio:
    """Workflow MCP stdio integration tests."""

    def test_workflow_round_trip_over_stdio(self, tmp_path: Path) -> None:
        """Run the MCP server over stdio and manage workflow records."""
        asyncio.run(self._run_workflow_round_trip(tmp_path))

    def test_agents_sdk_stdio_server_round_trip(
        self,
        tmp_path: Path,
    ) -> None:
        """Run workflow tools through the Agents SDK MCPServerStdio."""
        asyncio.run(self._run_agents_sdk_round_trip(tmp_path))

    def test_architect_mutations_persist_through_authorized_mcp(
        self,
        tmp_path: Path,
    ) -> None:
        """Persist only architect-authorized artifacts and tasks."""
        asyncio.run(self._run_architect_round_trip(tmp_path))

    async def _run_workflow_round_trip(self, tmp_path: Path) -> None:
        database_path = tmp_path / "workflow.db"
        environment = os.environ.copy()
        environment["AGENT_TEAM_DB_PATH"] = str(database_path)
        environment["PYTHONPATH"] = _pythonpath_with_src(environment)
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "agent_team.infrastructure.mcp.server.workflow_mcp_entrypoint",
            ],
            env=environment,
            cwd=Path.cwd(),
        )

        async with Client(
            stdio_client(parameters),
            mode="auto",
            read_timeout_seconds=10,
        ) as client:
            tools_result = await client.list_tools()
            tool_names = {tool.name for tool in tools_result.tools}
            assert {
                "create_feature",
                "get_feature",
                "get_feature_overview",
                "list_features",
                "add_artifact",
                "list_artifacts",
                "create_task",
                "list_tasks",
                "update_task_status",
            }.issubset(tool_names)

            created_feature_result = await client.call_tool(
                "create_feature",
                {
                    "title": "Build MCP server",
                    "description": "Store development work.",
                },
            )
            created_feature = _structured_content(created_feature_result)
            feature_id = created_feature["id"]
            assert isinstance(feature_id, int)

            retrieved_feature_result = await client.call_tool(
                "get_feature",
                {"feature_id": feature_id},
            )
            retrieved_feature = _structured_content(
                retrieved_feature_result,
            )
            assert retrieved_feature["title"] == "Build MCP server"

            created_artifact_result = await client.call_tool(
                "add_artifact",
                {
                    "feature_id": feature_id,
                    "kind": "requirements",
                    "content": "Store workflow records locally.",
                    "created_by": "agent:business_analyst",
                },
            )
            created_artifact = _structured_content(created_artifact_result)

            overview_result = await client.call_tool(
                "get_feature_overview",
                {"feature_id": feature_id},
            )
            overview = _structured_content(overview_result)
            overview_artifacts = overview["artifacts"]
            overview_tasks = overview["tasks"]
            assert isinstance(overview_artifacts, list)
            assert isinstance(overview_tasks, list)
            assert overview_artifacts == [created_artifact]
            assert overview_tasks == []

            created_task_result = await client.call_tool(
                "create_task",
                {
                    "feature_id": feature_id,
                    "title": "Implement repository",
                    "description": "Create SQLite persistence.",
                    "assigned_role": "backend_developer",
                },
            )
            created_task = _structured_content(created_task_result)
            task_id = created_task["id"]
            assert isinstance(task_id, int)

            listed_tasks_result = await client.call_tool(
                "list_tasks",
                {"feature_id": feature_id},
            )
            listed_tasks = _structured_list(listed_tasks_result)
            assert len(listed_tasks) == 1

            updated_task_result = await client.call_tool(
                "update_task_status",
                {
                    "task_id": task_id,
                    "status": "completed",
                },
            )
            updated_task = _structured_content(updated_task_result)
            assert updated_task["status"] == "completed"

    async def _run_agents_sdk_round_trip(self, tmp_path: Path) -> None:
        database_path = tmp_path / "workflow.db"
        environment = os.environ.copy()
        environment["AGENT_TEAM_DB_PATH"] = str(database_path)
        environment["PYTHONPATH"] = _pythonpath_with_src(environment)
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.DELIVERY_MANAGER,
        )
        authorizer = CapabilityAuthorizer(
            repository=workflow_repository_module.SQLiteWorkflowRepository(
                database_path
            ),
        )
        audit_repository = audit_repository_module.SQLiteAgentAuditRepository(
            database_path
        )
        run = audit_repository.start_run(
            AgentRunStart(
                role=DevelopmentRole.DELIVERY_MANAGER,
                model="qwen3.5:9b",
                prompt_hash="prompt-hash",
                prompt_excerpt="Create a feature.",
                max_turns=6,
            ),
        )
        server = workflow_mcp_factory.create_development_workflow_mcp_server(
            workflow_mcp_config.DevelopmentWorkflowMCPServerConfig(
                profile=profile,
                authorizer=authorizer,
                audit_repository=audit_repository,
                run=run,
            ),
            process_options=mcp_options.DevelopmentWorkflowMCPProcessOptions(
                environ=environment,
                python_executable=sys.executable,
                cwd=Path.cwd(),
            ),
        )

        await server.connect()
        try:
            tools = await server.list_tools()
            assert {tool.name for tool in tools} >= {
                "create_feature",
                "get_feature_overview",
                "create_task",
            }

            created_feature_result = await server.call_tool(
                "create_feature",
                {
                    "title": "Build Agents SDK MCP bridge",
                    "description": "Connect local agents to workflow tools.",
                },
            )
            created_feature = _structured_content(created_feature_result)
            first_content = created_feature_result.content[0]
            title = created_feature["title"]
            assert isinstance(title, str)
            assert isinstance(first_content, TextContent)
            assert title in first_content.text
        finally:
            await server.cleanup()

    async def _run_architect_round_trip(self, tmp_path: Path) -> None:
        database_path = tmp_path / "workflow.db"
        environment = os.environ.copy()
        environment["AGENT_TEAM_DB_PATH"] = str(database_path)
        environment["PYTHONPATH"] = _pythonpath_with_src(environment)
        workflow_repository = (
            workflow_repository_module.SQLiteWorkflowRepository(
                database_path,
            )
        )
        feature = workflow_repository.create_feature(
            title="Notifications",
            description="Alert users about important events.",
            status=FeatureStatus.DRAFT,
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.SOFTWARE_ARCHITECT,
        )
        authorizer = CapabilityAuthorizer(repository=workflow_repository)
        audit_repository = audit_repository_module.SQLiteAgentAuditRepository(
            database_path,
        )
        run = audit_repository.start_run(
            AgentRunStart(
                role=DevelopmentRole.SOFTWARE_ARCHITECT,
                model="qwen3.5:9b",
                prompt_hash="prompt-hash",
                prompt_excerpt="Save architecture.",
                max_turns=6,
                feature_id=feature.id,
            ),
        )
        server = workflow_mcp_factory.create_development_workflow_mcp_server(
            workflow_mcp_config.DevelopmentWorkflowMCPServerConfig(
                profile=profile,
                authorizer=authorizer,
                audit_repository=audit_repository,
                run=run,
            ),
            process_options=mcp_options.DevelopmentWorkflowMCPProcessOptions(
                environ=environment,
                python_executable=sys.executable,
                cwd=Path.cwd(),
            ),
        )

        await server.connect()
        try:
            tools = await server.list_tools()
            assert {tool.name for tool in tools} == {
                "add_artifact",
                "create_task",
                "get_feature",
                "get_feature_overview",
                "list_artifacts",
                "list_tasks",
            }

            architecture = _structured_content(
                await server.call_tool(
                    "add_artifact",
                    {
                        "feature_id": feature.id,
                        "kind": "architecture",
                        "content": "Use an event stream for alerts.",
                    },
                ),
            )
            implementation_plan = _structured_content(
                await server.call_tool(
                    "add_artifact",
                    {
                        "feature_id": feature.id,
                        "kind": "implementation_plan",
                        "content": "Backend first, then UI and QA.",
                    },
                ),
            )

            for assigned_role in (
                DevelopmentRole.BACKEND_DEVELOPER,
                DevelopmentRole.FRONTEND_DEVELOPER,
                DevelopmentRole.QA_ENGINEER,
                DevelopmentRole.CODE_REVIEWER,
            ):
                task = _structured_content(
                    await server.call_tool(
                        "create_task",
                        {
                            "feature_id": feature.id,
                            "title": f"{assigned_role.value} task",
                            "description": "Deliver the architected slice.",
                            "assigned_role": assigned_role.value,
                        },
                    ),
                )
                assert task["status"] == TaskStatus.PENDING.value

            denied_result = await server.call_tool(
                "create_task",
                {
                    "feature_id": feature.id,
                    "title": "Analyst task",
                    "description": "Should not persist.",
                    "assigned_role": "business_analyst",
                },
            )
        finally:
            await server.cleanup()

        assert denied_result.is_error is True
        artifacts = workflow_repository.list_artifacts(feature.id)
        tasks = workflow_repository.list_tasks(feature.id)
        assert [artifact.kind for artifact in artifacts] == [
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        ]
        assert {artifact.created_by for artifact in artifacts} == {
            "agent:software_architect",
        }
        assert architecture["created_by"] == "agent:software_architect"
        assert implementation_plan["created_by"] == (
            "agent:software_architect"
        )
        assert {task.assigned_role for task in tasks} == {
            DevelopmentRole.BACKEND_DEVELOPER,
            DevelopmentRole.FRONTEND_DEVELOPER,
            DevelopmentRole.QA_ENGINEER,
            DevelopmentRole.CODE_REVIEWER,
        }
        assert len(tasks) == 4


def _pythonpath_with_src(environment: Mapping[str, str]) -> str:
    source_path = str(Path.cwd() / "src")
    existing_path = environment.get("PYTHONPATH")
    if existing_path is None:
        return source_path
    return f"{source_path}{os.pathsep}{existing_path}"


def _structured_content(result: object) -> Mapping[str, object]:
    assert isinstance(result, CallToolResult)
    assert result.is_error is False
    content = result.structured_content
    assert isinstance(content, Mapping)
    return cast("Mapping[str, object]", content)


def _structured_list(result: object) -> list[object]:
    content = _structured_content(result)
    values = content["result"]
    assert isinstance(values, list)
    return cast("list[object]", values)
