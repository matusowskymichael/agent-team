"""Tests for the Ollama agent executor."""

import asyncio
from typing import Literal, cast

import httpx2
import pytest
from agents import Tool
from agents.mcp import MCPServer
from agents.memory import Session
from openai import APIConnectionError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.agent_runtime_instructions import (
    build_runtime_instructions,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.mcp.client import (
    workflow_mcp_unavailable_error,
)
from agent_team.infrastructure.ollama import ollama_agent_executor
from agent_team.infrastructure.ollama.ollama_agent_executor import (
    OllamaAgentExecutor,
)
from agent_team.infrastructure.ollama.ollama_model_factory import (
    create_ollama_model,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)
from tests.unit.fakes.audit.fake_agent_audit_repository import (
    FakeAgentAuditRepository,
)
from tests.unit.fakes.mcp.fake_mcp_server import FakeMCPServer
from tests.unit.fakes.runtime.fake_run_result import FakeRunResult

type FinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
]


class _Session:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _ModelResponse:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.usage = _Usage(input_tokens, output_tokens)


class TestOllamaAgentExecutor:
    """Ollama agent executor behavior tests."""

    def test_execute_disables_tracing_and_returns_final_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Disable tracing before running the local agent."""
        tracing_calls: list[bool] = []

        def record_tracing_disabled(disabled: bool) -> None:
            tracing_calls.append(disabled)

        async def run_agent(
            *_args: object,
            **_kwargs: object,
        ) -> FakeRunResult:
            return FakeRunResult(final_output="Local answer.")

        monkeypatch.setattr(
            ollama_agent_executor,
            "set_tracing_disabled",
            record_tracing_disabled,
        )
        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        settings = OllamaSettings()
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
        )
        profile = _profile()
        run = _run_record(profile.role)

        result = asyncio.run(
            executor.execute(
                AgentTask(prompt="Explain dependency inversion."),
                profile,
                run,
            ),
        )

        assert tracing_calls == [True]
        assert result.response == "Local answer."
        assert result.generation_metadata is not None
        assert result.generation_metadata.model == "qwen3.5:9b"
        assert result.generation_metadata.finish_reason is None

    def test_execute_maps_connection_error_to_ollama_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Map connection failures to concise Ollama availability errors."""

        async def run_agent(
            *_args: object,
            **_kwargs: object,
        ) -> FakeRunResult:
            request = httpx2.Request(
                method="POST",
                url="http://localhost:11434/v1/chat/completions",
            )
            raise APIConnectionError(request=request)

        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        settings = OllamaSettings()
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
        )
        profile = _profile()
        run = _run_record(profile.role)

        with pytest.raises(OllamaUnavailableError) as error:
            asyncio.run(
                executor.execute(
                    AgentTask(prompt="Hello."),
                    profile,
                    run,
                ),
            )

        assert settings.base_url in str(error.value)
        assert settings.model in str(error.value)

    def test_execute_attaches_mcp_server_and_manages_lifecycle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Connect MCP before the run and clean it up afterward."""
        events: list[str] = []
        factory_run_ids: list[int] = []
        server = FakeMCPServer(tool_names=[])
        profile = _profile()

        async def connect_server() -> None:
            events.append("connect")

        async def cleanup_server() -> None:
            events.append("cleanup")

        def record_tracing_disabled(disabled: bool) -> None:
            assert disabled is True
            events.append("trace")

        async def run_agent(
            starting_agent: object,
            prompt: object,
            **kwargs: object,
        ) -> FakeRunResult:
            assert isinstance(starting_agent, ollama_agent_executor.Agent)
            assert starting_agent.mcp_servers == [server]
            agent_values: dict[str, object] = vars(starting_agent)
            assert agent_values["instructions"] == build_runtime_instructions(
                profile,
            )
            assert prompt == "Create a feature."
            assert kwargs["max_turns"] == profile.run_limits.max_turns
            run_config = kwargs["run_config"]
            assert isinstance(run_config, ollama_agent_executor.RunConfig)
            assert run_config.tracing_disabled is True
            assert run_config.model_settings is not None
            assert run_config.model_settings.max_tokens == (
                settings.max_output_tokens
            )
            assert run_config.model_settings.extra_body == {"think": False}
            events.append("run")
            return FakeRunResult(final_output="Feature created.")

        monkeypatch.setattr(server, "connect", connect_server)
        monkeypatch.setattr(server, "cleanup", cleanup_server)
        monkeypatch.setattr(
            ollama_agent_executor,
            "set_tracing_disabled",
            record_tracing_disabled,
        )
        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        run = _run_record(profile.role)

        def create_servers(
            _profile: AgentProfile,
            received_run: AgentRunRecord,
            received_task: AgentTask,
        ) -> tuple[MCPServer, ...]:
            assert received_task.feature_id == 4
            factory_run_ids.append(received_run.id)
            return (server,)

        settings = OllamaSettings()
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
            mcp_server_factory=create_servers,
        )

        result = asyncio.run(
            executor.execute(
                AgentTask(prompt="Create a feature.", feature_id=4),
                profile,
                run,
            ),
        )

        assert result.response == "Feature created."
        assert events == ["connect", "trace", "run", "cleanup"]
        assert factory_run_ids == [run.id]

    def test_execute_passes_sdk_session_and_authoritative_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use the supplied local session and context instructions."""
        closeable_session = _Session()
        fake_session = cast("Session", closeable_session)
        received_contexts: list[AgentContextEnvelope] = []
        settings = OllamaSettings()
        profile = _profile()
        context = AgentContextEnvelope(
            feature_id=4,
            session_id="session-4",
            authoritative_context="AUTHORITATIVE WORKFLOW CONTEXT",
            max_conversation_history_items=3,
        )

        def create_session(
            received_context: AgentContextEnvelope,
        ) -> Session | None:
            received_contexts.append(received_context)
            return fake_session

        async def run_agent(
            starting_agent: object,
            prompt: object,
            **kwargs: object,
        ) -> FakeRunResult:
            assert isinstance(starting_agent, ollama_agent_executor.Agent)
            agent_values: dict[str, object] = vars(starting_agent)
            assert agent_values["instructions"] == build_runtime_instructions(
                profile,
                context,
            )
            assert prompt == "Summarize feature."
            assert kwargs["session"] is fake_session
            return FakeRunResult(final_output="Summary.")

        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
            session_factory=create_session,
        )
        run = _run_record(profile.role)

        result = asyncio.run(
            executor.execute(
                AgentTask(prompt="Summarize feature."),
                profile,
                run,
                context,
            ),
        )

        assert result.response == "Summary."
        assert received_contexts == [context]
        assert closeable_session.closed is True

    def test_execute_attaches_skill_tools_and_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Attach read-only skill tools beside workflow MCP servers."""
        settings = OllamaSettings()
        profile = _profile(DevelopmentRole.BUSINESS_ANALYST)
        run = _run_record(profile.role)
        skill_tool = cast("Tool", object())
        skill_context = "Available skills:\n- write-requirements-artifact"
        factory_calls: list[int] = []

        async def run_agent(
            starting_agent: object,
            prompt: object,
            **_kwargs: object,
        ) -> FakeRunResult:
            assert isinstance(starting_agent, ollama_agent_executor.Agent)
            agent_values: dict[str, object] = vars(starting_agent)
            assert agent_values["tools"] == [skill_tool]
            assert agent_values["instructions"] == build_runtime_instructions(
                profile,
                skill_context=skill_context,
            )
            assert prompt == "Add requirements."
            return FakeRunResult(final_output="Loaded skill.")

        def create_skill_tools(
            _profile: AgentProfile,
            received_run: AgentRunRecord,
        ) -> list[Tool]:
            factory_calls.append(received_run.id)
            return [skill_tool]

        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
            skill_tool_factory=create_skill_tools,
        )

        result = asyncio.run(
            executor.execute(
                AgentTask(prompt="Add requirements."),
                profile,
                run,
                skill_context=skill_context,
            ),
        )

        assert result.response == "Loaded skill."
        assert factory_calls == [run.id]

    def test_execute_attaches_workspace_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Attach developer workspace tools through the shared runtime."""
        settings = OllamaSettings()
        profile = _profile(DevelopmentRole.BACKEND_DEVELOPER)
        run = _run_record(profile.role)
        workspace_tool = cast("Tool", object())
        factory_task_ids: list[int | None] = []

        async def run_agent(
            starting_agent: object,
            prompt: object,
            **_kwargs: object,
        ) -> FakeRunResult:
            assert isinstance(starting_agent, ollama_agent_executor.Agent)
            agent_values: dict[str, object] = vars(starting_agent)
            assert agent_values["tools"] == [workspace_tool]
            assert prompt == "Patch backend."
            return FakeRunResult(final_output="Patched.")

        def create_workspace_tools(
            _profile: AgentProfile,
            _run: AgentRunRecord,
            received_task: AgentTask,
        ) -> list[Tool]:
            factory_task_ids.append(received_task.task_id)
            return [workspace_tool]

        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
            workspace_tool_factory=create_workspace_tools,
        )

        result = asyncio.run(
            executor.execute(
                AgentTask(
                    prompt="Patch backend.",
                    role=DevelopmentRole.BACKEND_DEVELOPER,
                    feature_id=1,
                    task_id=7,
                ),
                profile,
                run,
            ),
        )

        assert result.response == "Patched."
        assert factory_task_ids == [7]

    def test_execute_cleans_up_mcp_server_when_run_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clean up connected MCP servers after agent run failures."""
        events: list[str] = []
        server = FakeMCPServer(tool_names=[])
        profile = _profile()

        async def connect_server() -> None:
            events.append("connect")

        async def cleanup_server() -> None:
            events.append("cleanup")

        def record_tracing_disabled(disabled: bool) -> None:
            assert disabled is True
            events.append("trace")

        async def run_agent(
            starting_agent: object,
            prompt: object,
            **_kwargs: object,
        ) -> FakeRunResult:
            assert isinstance(starting_agent, ollama_agent_executor.Agent)
            assert prompt == "Create a feature."
            events.append("run")
            raise RuntimeError("Agent run failed.")

        monkeypatch.setattr(server, "connect", connect_server)
        monkeypatch.setattr(server, "cleanup", cleanup_server)
        monkeypatch.setattr(
            ollama_agent_executor,
            "set_tracing_disabled",
            record_tracing_disabled,
        )
        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        settings = OllamaSettings()
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
            mcp_server_factory=lambda _profile, _run, _task: (server,),
        )
        run = _run_record(profile.role)

        with pytest.raises(RuntimeError, match="Agent run failed"):
            asyncio.run(
                executor.execute(
                    AgentTask(prompt="Create a feature."),
                    profile,
                    run,
                ),
            )

        assert events == ["connect", "trace", "run", "cleanup"]

    def test_execute_maps_mcp_startup_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return a concise error when the MCP server cannot start."""
        server = FakeMCPServer(tool_names=[])
        profile = _profile()

        async def connect_server() -> None:
            raise RuntimeError("stdio unavailable")

        monkeypatch.setattr(server, "connect", connect_server)
        settings = OllamaSettings()
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
            mcp_server_factory=lambda _profile, _run, _task: (server,),
        )
        run = _run_record(profile.role)

        with pytest.raises(
            workflow_mcp_unavailable_error.WorkflowMCPUnavailableError
        ) as error:
            asyncio.run(
                executor.execute(
                    AgentTask(prompt="Create a feature."),
                    profile,
                    run,
                ),
            )

        assert str(error.value) == (
            "Development workflow MCP server could not start."
        )

    def test_execute_records_generation_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return finish reason and token usage without raw responses."""
        settings = OllamaSettings(model="qwen3.6:27b")
        model = create_ollama_model(settings)
        model.capture_finish_reason(_chat_completion("length"))

        async def run_agent(
            *_args: object,
            **_kwargs: object,
        ) -> FakeRunResult:
            return FakeRunResult(
                final_output="Partial answer.",
                raw_responses=(
                    _ModelResponse(input_tokens=10, output_tokens=20),
                    _ModelResponse(input_tokens=30, output_tokens=40),
                ),
            )

        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        executor = OllamaAgentExecutor(model=model, settings=settings)

        result = asyncio.run(
            executor.execute(
                AgentTask(prompt="Write proposal."),
                _profile(),
                _run_record(DevelopmentRole.DELIVERY_MANAGER),
            ),
        )

        assert result.response == "Partial answer."
        assert result.generation_metadata is not None
        assert result.generation_metadata.finish_reason == "length"
        assert result.generation_metadata.input_tokens == 40
        assert result.generation_metadata.output_tokens == 60
        assert result.generation_metadata.visible_output_char_count == 15
        assert result.generation_metadata.objectively_truncated is True

    def test_execute_omits_hidden_reasoning_from_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Never return hidden reasoning text for CLI printing or audit."""

        async def run_agent(
            *_args: object,
            **_kwargs: object,
        ) -> FakeRunResult:
            return FakeRunResult(
                final_output="<think>hidden chain</think>Visible answer.",
            )

        monkeypatch.setattr(
            ollama_agent_executor.Runner,
            "run",
            run_agent,
        )
        settings = OllamaSettings()
        executor = OllamaAgentExecutor(
            model=create_ollama_model(settings),
            settings=settings,
        )

        result = asyncio.run(
            executor.execute(
                AgentTask(prompt="Answer."),
                _profile(),
                _run_record(DevelopmentRole.DELIVERY_MANAGER),
            ),
        )

        assert "hidden chain" not in result.response
        assert result.response == "[hidden reasoning omitted]Visible answer."


def _profile(
    role: DevelopmentRole = DevelopmentRole.DELIVERY_MANAGER,
) -> AgentProfile:
    return AgentProfileCatalog().get_profile(role)


def _run_record(role: DevelopmentRole) -> AgentRunRecord:
    audit_repository = FakeAgentAuditRepository()
    return audit_repository.open_run(role=role)


def _chat_completion(finish_reason: FinishReason) -> ChatCompletion:
    return ChatCompletion(
        id="completion-id",
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionMessage(
                    content="Partial answer.",
                    role="assistant",
                ),
            ),
        ],
        created=1,
        model="qwen3.6:27b",
        object="chat.completion",
    )
