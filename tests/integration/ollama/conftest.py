"""Offline local model fixtures for real SDK execution segments."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from agents import Tool, function_tool
from agents.items import ModelResponse, TResponseInputItem

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
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
from tests.integration.ollama.sdk_segment_scenario import SdkSegmentScenario
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
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
    return SdkSegmentScenario(
        executor=OllamaAgentExecutor(
            model=model,
            settings=settings,
            workspace_tool_factory=workspace_tools,
            session_factory=sessions.create_session,
        ),
        session_factory=sessions,
        profile=profile,
        run=FakeAgentAuditRepository().open_run(task, profile.role),
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
    )
