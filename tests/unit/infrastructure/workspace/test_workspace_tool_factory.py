"""Tests for Agents SDK restricted workspace tools."""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from agents import FunctionTool, Tool
from agents.tool_context import ToolContext

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.workspace.workspace_service import WorkspaceService
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.code_search_result import CodeSearchResult
from agent_team.domain.workspace.patch_application_result import (
    PatchApplicationResult,
)
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_file_content import (
    WorkspaceFileContent,
)
from agent_team.domain.workspace.workspace_file_listing import (
    WorkspaceFileListing,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)
from agent_team.infrastructure.workspace.workspace_tool_factory import (
    WORKSPACE_SERVER_NAME,
    WorkspaceToolFactory,
)
from tests.reporting.allure_steps import report_step
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


class TestWorkspaceToolFactory:
    """Workspace tool factory behavior tests."""

    def test_creates_tools_for_developer_profiles(
        self,
        tmp_path: Path,
    ) -> None:
        """Expose workspace tools only to backend and frontend profiles."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        factory = _factory(repository, audit_repository)

        backend_tools = factory.create_tools(
            AgentProfileCatalog().get_profile(
                DevelopmentRole.BACKEND_DEVELOPER
            ),
            audit_repository.open_run(role=DevelopmentRole.BACKEND_DEVELOPER),
            _task(DevelopmentRole.BACKEND_DEVELOPER, tmp_path),
        )
        analyst_tools = factory.create_tools(
            AgentProfileCatalog().get_profile(
                DevelopmentRole.BUSINESS_ANALYST
            ),
            audit_repository.open_run(role=DevelopmentRole.BUSINESS_ANALYST),
            AgentTask(prompt="No workspace."),
        )

        assert _tool_names(backend_tools) == {
            "apply_patch",
            "list_files",
            "read_file",
            "run_check",
            "search_code",
        }
        assert analyst_tools == []

    def test_read_file_audits_hashes(self, tmp_path: Path) -> None:
        """Audit read metadata without retaining complete source content."""
        _write(tmp_path, "backend/auth.py", "API_TOKEN = 'secret'\n")
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        tool = _tool(
            "read_file",
            repository,
            audit_repository,
            DevelopmentRole.BACKEND_DEVELOPER,
            tmp_path,
        )

        result = _invoke(tool, '{"path":"backend/auth.py"}')

        assert result["content"] == "API_TOKEN = 'secret'\n"
        invocation = audit_repository.tool_invocations[1]
        assert invocation.server_name == WORKSPACE_SERVER_NAME
        assert invocation.status is ToolInvocationStatus.COMPLETED
        assert "API_TOKEN" not in (invocation.result_preview or "")
        assert "secret" not in (invocation.result_preview or "")
        assert "content_hash" in (invocation.result_preview or "")

    def test_list_and_search_tools_audit_results(
        self,
        tmp_path: Path,
    ) -> None:
        """Audit completed read-only workspace tool invocations."""
        _write(tmp_path, "backend/auth.py", "class AuthService:\n    pass\n")
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        list_tool = _tool(
            "list_files",
            repository,
            audit_repository,
            DevelopmentRole.BACKEND_DEVELOPER,
            tmp_path,
        )
        search_tool = _tool(
            "search_code",
            repository,
            audit_repository,
            DevelopmentRole.BACKEND_DEVELOPER,
            tmp_path,
        )

        listing = _invoke(list_tool, '{"directory":""}')
        search = _invoke(search_tool, '{"query":"AuthService"}')

        assert listing["files"] == ["backend/auth.py"]
        assert search["matches"] == [
            {
                "path": "backend/auth.py",
                "line_number": 1,
                "line_excerpt": "class AuthService:",
            },
        ]
        completed = [
            invocation
            for invocation in audit_repository.tool_invocations.values()
            if invocation.status is ToolInvocationStatus.COMPLETED
        ]
        assert [invocation.tool_name for invocation in completed] == [
            "list_files",
            "search_code",
        ]

    def test_patch_audit_omits_raw_patch_text(self, tmp_path: Path) -> None:
        """Audit patch hashes and path without storing raw source patches."""
        _write(tmp_path, "backend/auth.py", "return False\n")
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        tool = _tool(
            "apply_patch",
            repository,
            audit_repository,
            DevelopmentRole.BACKEND_DEVELOPER,
            tmp_path,
        )

        result = _invoke(
            tool,
            (
                '{"path":"backend/auth.py","old_text":"return False",'
                '"new_text":"return True"}'
            ),
        )

        assert result["applied"] is True
        invocation = audit_repository.tool_invocations[1]
        assert '"path":"backend/auth.py"' in invocation.arguments_preview_json
        assert "return False" not in invocation.arguments_preview_json
        assert "return True" not in invocation.arguments_preview_json
        assert "old_text_hash" in invocation.arguments_preview_json
        assert "new_text_hash" in invocation.arguments_preview_json

    def test_denied_patch_does_not_modify_workspace(
        self,
        tmp_path: Path,
    ) -> None:
        """Deny unauthorized workspace mutation before executor execution."""
        with report_step("Arrange a role-restricted workspace mutation"):
            _write(
                tmp_path,
                "frontend/LoginForm.tsx",
                "export const A = 1\n",
            )
            repository = _repository_with_task(
                DevelopmentRole.BACKEND_DEVELOPER,
            )
            audit_repository = FakeAgentAuditRepository()
            tool = _tool(
                "apply_patch",
                repository,
                audit_repository,
                DevelopmentRole.BACKEND_DEVELOPER,
                tmp_path,
            )

        with report_step("Attempt the denied workspace patch"):
            result = _invoke(
                tool,
                (
                    '{"path":"frontend/LoginForm.tsx","old_text":"1",'
                    '"new_text":"2"}'
                ),
            )

        with report_step("Verify denial and unchanged workspace content"):
            assert "WORKSPACE_CAPABILITY_DENIED" in str(result["error"])
            assert (tmp_path / "frontend/LoginForm.tsx").read_text() == (
                "export const A = 1\n"
            )
            invocation = audit_repository.tool_invocations[1]
            assert invocation.status is ToolInvocationStatus.DENIED

    def test_audit_start_failure_prevents_workspace_execution(
        self,
        tmp_path: Path,
    ) -> None:
        """Prevent workspace execution when audit start fails."""
        _write(tmp_path, "backend/auth.py", "return False\n")
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository(
            fail_start_tool_invocation=True,
        )
        tool = _tool(
            "apply_patch",
            repository,
            audit_repository,
            DevelopmentRole.BACKEND_DEVELOPER,
            tmp_path,
        )

        with pytest.raises(RuntimeError, match="tool audit start failed"):
            _invoke(
                tool,
                (
                    '{"path":"backend/auth.py","old_text":"False",'
                    '"new_text":"True"}'
                ),
            )

        assert (tmp_path / "backend/auth.py").read_text() == "return False\n"

    def test_run_check_uses_role_allowlist(self, tmp_path: Path) -> None:
        """Run only checks exposed by the immutable profile."""
        repository = _repository_with_task(DevelopmentRole.FRONTEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        tool = _tool(
            "run_check",
            repository,
            audit_repository,
            DevelopmentRole.FRONTEND_DEVELOPER,
            tmp_path,
        )

        result = _invoke(tool, '{"name":"frontend"}')

        assert result["exit_code"] == 0
        assert result["name"] == "frontend"

    def test_missing_workspace_root_is_denied_before_service_creation(
        self,
    ) -> None:
        """Deny workspace tools before service creation without a root."""
        audit_repository = FakeAgentAuditRepository()

        def fail_if_called(_workspace_root: Path) -> WorkspaceService:
            raise AssertionError("service factory should not be called")

        factory = WorkspaceToolFactory(
            service_factory=fail_if_called,
            audit_repository=audit_repository,
        )
        profile = AgentProfileCatalog().get_profile(
            DevelopmentRole.BACKEND_DEVELOPER,
        )
        run = audit_repository.open_run(
            role=DevelopmentRole.BACKEND_DEVELOPER,
            feature_id=1,
        )
        tools = factory.create_tools(
            profile,
            run,
            AgentTask(
                prompt="List files.",
                role=DevelopmentRole.BACKEND_DEVELOPER,
                feature_id=1,
                task_id=1,
            ),
        )
        tool = next(cast("FunctionTool", item) for item in tools)

        result = _invoke(tool, '{"directory":""}')

        assert "WORKSPACE_CAPABILITY_DENIED" in str(result["error"])
        assert audit_repository.start_tool_invocation_calls == 0
        assert audit_repository.tool_invocations[1].status is (
            ToolInvocationStatus.DENIED
        )

    @pytest.mark.parametrize(
        ("tool_name", "arguments_json"),
        [
            ("run_check", "{}"),
            ("list_files", '{"directory":42}'),
        ],
    )
    def test_invalid_tool_arguments_are_non_retryable_denials(
        self,
        tmp_path: Path,
        tool_name: str,
        arguments_json: str,
    ) -> None:
        """Return explicit denials for invalid model-supplied arguments."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        tool = _tool(
            tool_name,
            repository,
            audit_repository,
            DevelopmentRole.BACKEND_DEVELOPER,
            tmp_path,
        )

        result = _invoke(tool, arguments_json)

        assert "WORKSPACE_CAPABILITY_DENIED" in str(result["error"])
        assert "Do not retry" in str(result["error"])
        assert audit_repository.start_tool_invocation_calls == 0
        assert audit_repository.tool_invocations[1].status is (
            ToolInvocationStatus.DENIED
        )

    def test_non_object_arguments_raise_before_execution(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject non-object JSON tool arguments."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        tool = _tool(
            "list_files",
            repository,
            audit_repository,
            DevelopmentRole.BACKEND_DEVELOPER,
            tmp_path,
        )

        with pytest.raises(WorkspaceAccessDeniedError, match="object"):
            _invoke(tool, "[]")

        assert audit_repository.start_tool_invocation_calls == 0

    def test_executor_denial_after_audit_start_is_recorded(
        self,
        tmp_path: Path,
    ) -> None:
        """Record executor denials that occur after initial authorization."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        tool = _tool(
            "read_file",
            repository,
            audit_repository,
            DevelopmentRole.BACKEND_DEVELOPER,
            tmp_path,
        )

        result = _invoke(tool, '{"path":"backend/missing.py"}')

        assert "WORKSPACE_CAPABILITY_DENIED" in str(result["error"])
        invocation = audit_repository.tool_invocations[1]
        assert invocation.status is ToolInvocationStatus.FAILED
        assert invocation.error_type == "WorkspaceAccessDeniedError"

    def test_unexpected_executor_failure_is_recorded(
        self,
        tmp_path: Path,
    ) -> None:
        """Record unexpected workspace execution failures distinctly."""
        repository = _repository_with_task(DevelopmentRole.BACKEND_DEVELOPER)
        audit_repository = FakeAgentAuditRepository()
        factory = WorkspaceToolFactory(
            service_factory=lambda _root: WorkspaceService(
                repository=repository,
                executor=_ExplodingWorkspaceExecutor(),
            ),
            audit_repository=audit_repository,
        )
        run = audit_repository.open_run(
            role=DevelopmentRole.BACKEND_DEVELOPER,
            feature_id=1,
        )
        tool = next(
            cast("FunctionTool", item)
            for item in factory.create_tools(
                AgentProfileCatalog().get_profile(
                    DevelopmentRole.BACKEND_DEVELOPER,
                ),
                run,
                _task(DevelopmentRole.BACKEND_DEVELOPER, tmp_path),
            )
            if item.name == "list_files"
        )

        result = _invoke(tool, '{"directory":""}')

        assert result["error_type"] == "RuntimeError"
        assert "WORKSPACE_TOOL_FAILED" in str(result["error"])
        invocation = audit_repository.tool_invocations[1]
        assert invocation.status is ToolInvocationStatus.FAILED
        assert invocation.error_type == "RuntimeError"


def _factory(
    repository: FakeWorkflowRepository,
    audit_repository: FakeAgentAuditRepository,
) -> WorkspaceToolFactory:
    return WorkspaceToolFactory(
        service_factory=lambda workspace_root: WorkspaceService(
            repository=repository,
            executor=LocalWorkspaceExecutor(
                workspace_root,
                check_commands={
                    "backend": ("python", "-c", "print('backend-ok')"),
                    "frontend": ("python", "-c", "print('frontend-ok')"),
                },
            ),
        ),
        audit_repository=audit_repository,
    )


def _tool(
    name: str,
    repository: FakeWorkflowRepository,
    audit_repository: FakeAgentAuditRepository,
    role: DevelopmentRole,
    workspace_root: Path,
) -> FunctionTool:
    run = audit_repository.open_run(role=role, feature_id=1)
    tools = _factory(repository, audit_repository).create_tools(
        AgentProfileCatalog().get_profile(role),
        run,
        _task(role, workspace_root),
    )
    return next(
        cast("FunctionTool", tool) for tool in tools if tool.name == name
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


def _tool_names(tools: list[Tool]) -> set[str]:
    return {tool.name for tool in tools}


def _repository_with_task(
    assigned_role: DevelopmentRole,
) -> FakeWorkflowRepository:
    repository = FakeWorkflowRepository()
    feature = repository.create_feature(
        title="Feature",
        description="Description.",
        status=FeatureStatus.DRAFT,
    )
    repository.create_task(
        feature_id=feature.id,
        title="Task",
        description="Task description.",
        assigned_role=assigned_role,
        status=TaskStatus.PENDING,
    )
    return repository


def _task(role: DevelopmentRole, workspace_root: Path) -> AgentTask:
    return AgentTask(
        prompt="Patch code.",
        role=role,
        feature_id=1,
        task_id=1,
        workspace_root=workspace_root,
    )


def _write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _ExplodingWorkspaceExecutor:
    """Workspace executor fake that fails during execution."""

    def list_files(self, directory: str = "") -> WorkspaceFileListing:
        raise RuntimeError(f"boom {directory}")

    def search_code(self, query: str) -> CodeSearchResult:
        raise RuntimeError(f"boom {query}")

    def read_file(self, path: str) -> WorkspaceFileContent:
        raise RuntimeError(f"boom {path}")

    def apply_patch(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ) -> PatchApplicationResult:
        raise RuntimeError(f"boom {path} {old_text} {new_text}")

    def run_check(self, name: str) -> CheckRunResult:
        raise RuntimeError(f"boom {name}")
