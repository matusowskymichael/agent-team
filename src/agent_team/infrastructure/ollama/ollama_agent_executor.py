"""Agent executor backed by local Ollama."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from agents import Agent, Model, RunConfig, Runner, Tool, set_tracing_disabled
from agents.mcp import MCPServer
from openai import APIConnectionError, APITimeoutError

from agent_team.application.audit.audit_sanitizer import (
    omit_hidden_reasoning,
)
from agent_team.application.runtime.agent_runtime_instructions import (
    build_runtime_instructions,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)

from ..mcp.client.workflow_mcp_unavailable_error import (
    WorkflowMCPUnavailableError,
)
from ..persistence.sqlite.sessions.sqlite_session_factory import (
    SessionFactory,
    close_session,
    no_session,
)
from .ollama_chat_completions_model import metadata_from_model
from .ollama_model_settings import create_ollama_model_settings

AGENT_NAME = "Local development workflow coordinator"


def _no_mcp_servers(
    _profile: AgentProfile,
    _run: AgentRunRecord,
    _task: AgentTask,
) -> tuple[MCPServer, ...]:
    return ()


def _no_skill_tools(
    _profile: AgentProfile,
    _run: AgentRunRecord,
) -> list[Tool]:
    return []


def _no_workspace_tools(
    _profile: AgentProfile,
    _run: AgentRunRecord,
    _task: AgentTask,
) -> list[Tool]:
    return []


@dataclass(frozen=True, slots=True)
class OllamaAgentExecutor:
    """An agent runtime that runs prompts against local Ollama."""

    model: Model
    settings: OllamaSettings
    mcp_server_factory: Callable[
        [AgentProfile, AgentRunRecord, AgentTask],
        tuple[MCPServer, ...],
    ] = _no_mcp_servers
    skill_tool_factory: Callable[
        [AgentProfile, AgentRunRecord],
        list[Tool],
    ] = _no_skill_tools
    workspace_tool_factory: Callable[
        [AgentProfile, AgentRunRecord, AgentTask],
        list[Tool],
    ] = _no_workspace_tools
    session_factory: SessionFactory = no_session

    @property
    def model_name(self) -> str:
        """Return the configured Ollama model name."""
        return self.settings.model

    async def execute(
        self,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
        context: AgentContextEnvelope | None = None,
        skill_context: str | None = None,
    ) -> AgentResult:
        """Execute an agent task using the Agents SDK Runner."""
        connected_servers: list[MCPServer] = []
        mcp_servers = self.mcp_server_factory(profile, run, task)
        skill_tools = self.skill_tool_factory(profile, run)
        workspace_tools = self.workspace_tool_factory(profile, run, task)
        session = None if context is None else self.session_factory(context)

        try:
            for mcp_server in mcp_servers:
                await _connect_mcp_server(mcp_server)
                connected_servers.append(mcp_server)

            set_tracing_disabled(True)
            agent = Agent(
                name=AGENT_NAME,
                instructions=build_runtime_instructions(
                    profile,
                    context,
                    skill_context,
                    task,
                ),
                model=self.model,
                tools=[*skill_tools, *workspace_tools],
                mcp_servers=list(mcp_servers),
            )
            result = await Runner.run(
                agent,
                task.prompt,
                max_turns=profile.run_limits.max_turns,
                run_config=RunConfig(
                    tracing_disabled=True,
                    model_settings=create_ollama_model_settings(
                        self.settings,
                    ),
                ),
                session=session,
            )
        except (APIConnectionError, APITimeoutError) as error:
            message = (
                f"Ollama is unavailable at {self.settings.base_url}. "
                f"Start Ollama and ensure {self.settings.model} is available."
            )
            raise OllamaUnavailableError(message) from error
        finally:
            for mcp_server in reversed(connected_servers):
                await mcp_server.cleanup()
            close_session(session)

        final_output: object = result.final_output
        response = omit_hidden_reasoning(str(final_output))
        input_tokens, output_tokens = _usage_tokens(result)
        return AgentResult(
            response=response,
            generation_metadata=metadata_from_model(
                model=self.model,
                model_name=self.settings.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                visible_output=response,
            ),
        )


async def _connect_mcp_server(mcp_server: MCPServer) -> None:
    try:
        await mcp_server.connect()
    except Exception as error:
        message = "Development workflow MCP server could not start."
        raise WorkflowMCPUnavailableError(message) from error


def _usage_tokens(result: object) -> tuple[int | None, int | None]:
    raw_responses_value = getattr(result, "raw_responses", None)
    if not isinstance(raw_responses_value, list | tuple):
        return None, None
    raw_responses = cast(
        "list[object] | tuple[object, ...]",
        raw_responses_value,
    )
    if not raw_responses:
        return None, None
    input_tokens = 0
    output_tokens = 0
    for response in raw_responses:
        usage = getattr(response, "usage", None)
        input_tokens += _token_count(getattr(usage, "input_tokens", None))
        output_tokens += _token_count(getattr(usage, "output_tokens", None))
    return input_tokens, output_tokens


def _token_count(value: object) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return 0
