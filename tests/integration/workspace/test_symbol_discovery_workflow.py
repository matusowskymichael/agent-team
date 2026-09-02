"""Integration tests for developer symbol discovery."""

import asyncio
from pathlib import Path
from typing import Any, cast

from agents import FunctionTool, Tool
from agents.tool_context import ToolContext

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.workspace.workspace_service import WorkspaceService
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)
from agent_team.infrastructure.workspace.workspace_tool_factory import (
    WorkspaceToolFactory,
)

# Any is required by the installed Agents SDK FunctionTool callable boundary.


class TestSymbolDiscoveryWorkflow:
    """Developer workspace symbol-discovery integration tests."""

    def test_existing_method_is_read_without_duplicate_mutation(
        self,
        tmp_path: Path,
    ) -> None:
        """Discover and inspect an existing method without applying a patch."""
        database_path = tmp_path / "workflow.db"
        workspace_root = tmp_path / "workspace"
        source_path = workspace_root / "backend" / "password_reset.py"
        source_path.parent.mkdir(parents=True)
        source = (
            "class PasswordResetService:\n"
            "    def generate_token(self, user_id: int) -> str:\n"
            "        return f'reset-{user_id}'\n"
        )
        source_path.write_text(source, encoding="utf-8")
        test_path = workspace_root / "tests" / "test_password_reset.py"
        test_path.parent.mkdir(parents=True)
        test_source = (
            "def test_generate_token() -> None:\n"
            "    assert PasswordResetService().generate_token(1)\n"
        )
        test_path.write_text(test_source, encoding="utf-8")

        workflow_repository = (
            workflow_repository_module.SQLiteWorkflowRepository(
                database_path,
            )
        )
        feature = workflow_repository.create_feature(
            title="Password Reset",
            description="Reset account passwords.",
            status=FeatureStatus.IMPLEMENTATION,
        )
        task = workflow_repository.create_task(
            feature_id=feature.id,
            title="Reuse token generation",
            description="Reuse the existing token method.",
            assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
            status=TaskStatus.PENDING,
        )
        audit_repository = audit_repository_module.SQLiteAgentAuditRepository(
            database_path,
        )
        run = audit_repository.start_run(
            AgentRunStart(
                role=DevelopmentRole.BACKEND_DEVELOPER,
                model="test-model",
                prompt_hash="prompt-hash",
                prompt_excerpt="Reuse token generation.",
                max_turns=10,
                feature_id=feature.id,
            ),
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        tools = WorkspaceToolFactory(
            service_factory=lambda root: WorkspaceService(
                repository=workflow_repository,
                executor=LocalWorkspaceExecutor(root),
            ),
            audit_repository=audit_repository,
        ).create_tools(
            profile,
            run,
            AgentTask(
                prompt="Reuse the existing token method.",
                role=DevelopmentRole.BACKEND_DEVELOPER,
                feature_id=feature.id,
                task_id=task.id,
                workspace_root=workspace_root,
            ),
        )

        found = _invoke(
            _tool(tools, "find_symbol"),
            '{"name":"PasswordResetService.generate_token"}',
        )
        read = _invoke(
            _tool(tools, "read_file"),
            '{"path":"backend/password_reset.py"}',
        )
        nearby_tests = _invoke(
            _tool(tools, "search_code"),
            '{"query":"generate_token"}',
        )
        read_test = _invoke(
            _tool(tools, "read_file"),
            '{"path":"tests/test_password_reset.py"}',
        )

        assert found["definitions"] == [
            {
                "path": "backend/password_reset.py",
                "line_number": 2,
                "name": "generate_token",
                "qualified_name": "PasswordResetService.generate_token",
                "kind": "method",
            },
        ]
        assert read["content"] == source
        matches = cast(
            "list[dict[str, object]]",
            nearby_tests["matches"],
        )
        assert any(
            match["path"] == "tests/test_password_reset.py"
            for match in matches
        )
        assert read_test["content"] == test_source
        assert source_path.read_text(encoding="utf-8") == source
        invocations = audit_repository.list_tool_invocations(run.id)
        assert [invocation.tool_name for invocation in invocations] == [
            "find_symbol",
            "read_file",
            "search_code",
            "read_file",
        ]


def _tool(tools: list[Tool], name: str) -> FunctionTool:
    return next(
        cast("FunctionTool", tool)
        for tool in tools
        if getattr(tool, "name", None) == name
    )


def _invoke(tool: FunctionTool, arguments_json: str) -> dict[str, object]:
    result = asyncio.run(
        tool.on_invoke_tool(
            cast("ToolContext[Any]", object()),
            arguments_json,
        ),
    )
    assert isinstance(result, dict)
    return cast("dict[str, object]", result)
