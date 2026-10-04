"""Offline local model fixtures for real SDK execution segments."""

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from agents import Tool, function_tool
from agents.items import ModelResponse, TResponseInputItem
from agents.mcp import MCPServerStdio

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.workspace.workspace_service import WorkspaceService
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.infrastructure.mcp.client.owned_workflow_mcp_server import (
    OwnedWorkflowMCPServer,
)
from agent_team.infrastructure.ollama.ollama_agent_executor import (
    OllamaAgentExecutor,
)
from agent_team.infrastructure.ollama.ollama_model_factory import (
    create_ollama_model,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_session_factory,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)
from agent_team.infrastructure.workspace.workspace_tool_factory import (
    WorkspaceToolFactory,
)
from tests.integration.ollama.sdk_segment_scenario import SdkSegmentScenario
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.workflow.fake_workflow_repository import (
    FakeWorkflowRepository,
)


@pytest.fixture
def sdk_segment_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SdkSegmentScenario:
    """Replace only model inference while exercising the installed SDK."""
    responses: list[ModelResponse | BaseException] = []
    inputs: list[str | list[TResponseInputItem]] = []
    operations: list[str] = []

    async def get_response(
        *_args: object,
        **kwargs: object,
    ) -> ModelResponse:
        inputs.append(cast("str | list[TResponseInputItem]", kwargs["input"]))
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    @function_tool
    def apply_patch() -> str:
        """Record one successful workspace mutation."""
        operations.append("apply_patch")
        return "Patch applied."

    @function_tool
    def run_check() -> str:
        """Record one completed trusted check."""
        operations.append("run_check")
        return "Check passed."

    def workspace_tools(
        _profile: AgentProfile,
        _run: AgentRunRecord,
        _task: AgentTask,
    ) -> list[Tool]:
        return [apply_patch, run_check]

    settings = OllamaSettings()
    model = create_ollama_model(settings)
    monkeypatch.setattr(model, "get_response", get_response)
    sessions = sqlite_session_factory.SQLiteSessionFactory(
        tmp_path / "sessions.db",
    )
    profile = replace(
        AgentProfileCatalog().get_profile(DevelopmentRole.BACKEND_DEVELOPER),
        run_limits=AgentRunLimits(segment_turns=2),
    )
    task = AgentTask(
        prompt="Implement the bound task and submit a verified handoff.",
        role=profile.role,
        feature_id=4,
        task_id=9,
        session_id="feature-4-backend",
        workspace_root=tmp_path,
    )
    audit = FakeAgentAuditRepository()
    return SdkSegmentScenario(
        executor=OllamaAgentExecutor(
            model=model,
            settings=settings,
            workspace_tool_factory=workspace_tools,
            session_factory=sessions.create_session,
        ),
        session_factory=sessions,
        profile=profile,
        run=audit.open_run(task, profile.role),
        task=task,
        context=AgentContextEnvelope(
            feature_id=4,
            session_id="feature-4-backend",
            authoritative_context="Bound task 9 is in progress.",
            max_conversation_history_items=20,
        ),
        responses=responses,
        inputs=inputs,
        operations=operations,
        audit=audit,
    )


@pytest.fixture(
    params=[OSError, asyncio.CancelledError],
    ids=["infrastructure", "cancelled-tool"],
)
def sdk_workspace_failure_scenario(
    sdk_segment_scenario: SdkSegmentScenario,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> SdkSegmentScenario:
    """Bind real restricted workspace tools and a failing check to the SDK."""
    scenario = sdk_segment_scenario
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "auth.py").write_text("return 0\n", encoding="utf-8")
    repository = FakeWorkflowRepository()
    feature = repository.create_feature(
        "Logout", "Implement the assigned task.", FeatureStatus.DRAFT
    )
    development_task = repository.create_task(
        feature.id,
        "Revise logout",
        "Implement logout.",
        DevelopmentRole.BACKEND_DEVELOPER,
        TaskStatus.IN_PROGRESS,
    )
    task = replace(
        scenario.task,
        feature_id=feature.id,
        task_id=development_task.id,
    )
    audit = FakeAgentAuditRepository()
    error_type = cast("type[BaseException]", request.param)
    error = error_type("Trusted check infrastructure failed.")
    workspace = WorkspaceToolFactory(
        service_factory=lambda root: WorkspaceService(
            repository=repository,
            executor=LocalWorkspaceExecutor(root),
        ),
        audit_repository=audit,
    )

    def fail_workspace_check(
        _executor: LocalWorkspaceExecutor,
        name: str,
    ) -> CheckRunResult:
        assert name == "backend"
        assert (backend / "auth.py").read_text(encoding="utf-8") == (
            "return 1\n"
        )
        raise error

    monkeypatch.setattr(
        LocalWorkspaceExecutor, "run_check", fail_workspace_check
    )
    return replace(
        scenario,
        task=task,
        context=replace(
            scenario.context,
            feature_id=feature.id,
            task_id=development_task.id,
            authoritative_context="Bound task 1 is in progress.",
        ),
        executor=replace(
            scenario.executor,
            workspace_tool_factory=workspace.create_tools,
        ),
        audit=audit,
        repository=repository,
        workspace_error=error,
    )


@pytest.fixture
def sdk_mcp_watchdog_scenario(
    sdk_segment_scenario: SdkSegmentScenario,
    tmp_path: Path,
) -> SdkSegmentScenario:
    """Run an owned real MCP process with offline SDK model inference."""
    environment = dict(os.environ)
    environment["AGENT_TEAM_DB_PATH"] = str(tmp_path / "workflow.db")
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    server = OwnedWorkflowMCPServer(
        params={
            "command": sys.executable,
            "args": [
                "-c",
                "import os,pathlib; "
                "pathlib.Path('mcp.pid').write_text(str(os.getpid())); "
                "from agent_team.infrastructure.mcp.server."
                "workflow_mcp_entrypoint import main; main()",
            ],
            "env": environment,
            "cwd": tmp_path,
        },
        client_session_timeout_seconds=10,
    )

    def servers(
        _profile: AgentProfile,
        _run: AgentRunRecord,
        _task: AgentTask,
    ) -> tuple[MCPServerStdio, ...]:
        return (server,)

    return replace(
        sdk_segment_scenario,
        executor=replace(
            sdk_segment_scenario.executor,
            mcp_server_factory=servers,
        ),
    )


@pytest.fixture
def sdk_workspace_cancellation_scenario(
    sdk_segment_scenario: SdkSegmentScenario,
    tmp_path: Path,
) -> SdkSegmentScenario:
    """Bind a trusted cancellable workspace command to the real SDK."""
    scenario = sdk_segment_scenario
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "auth.py").write_text("return 0\n", encoding="utf-8")
    repository = FakeWorkflowRepository()
    feature = repository.create_feature(
        "Logout",
        "Implement the assigned task.",
        FeatureStatus.DRAFT,
    )
    development_task = repository.create_task(
        feature.id,
        "Revise logout",
        "Implement logout.",
        DevelopmentRole.BACKEND_DEVELOPER,
        TaskStatus.IN_PROGRESS,
    )
    task = replace(
        scenario.task, feature_id=feature.id, task_id=development_task.id
    )
    workspace = WorkspaceToolFactory(
        service_factory=lambda root: WorkspaceService(
            repository=repository,
            executor=LocalWorkspaceExecutor(
                root,
                check_commands={
                    "backend": (
                        "python",
                        "-c",
                        "import os,time,pathlib; "
                        "pathlib.Path('check.pid').write_text("
                        "str(os.getpid())); "
                        "time.sleep(0.2)",
                    ),
                },
            ),
        ),
        audit_repository=scenario.audit,
    )
    return replace(
        scenario,
        task=task,
        repository=repository,
        executor=replace(
            scenario.executor, workspace_tool_factory=workspace.create_tools
        ),
    )
