"""Agent executor backed by local Ollama."""

import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from agents import (
    Agent,
    MaxTurnsExceeded,
    Model,
    RunConfig,
    Runner,
    Tool,
    set_tracing_disabled,
)
from agents.items import TResponseInputItem
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
from agent_team.domain.runtime.agent_cleanup_budget import AgentCleanupBudget
from agent_team.domain.runtime.agent_cleanup_timeout_error import (
    AgentCleanupTimeoutError,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_provider_timeout_error import (
    AgentProviderTimeoutError,
)
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_segment_timeout_error import (
    AgentSegmentTimeoutError,
)
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)

from ..mcp.client.mcp_process_cleanup import MCPProcessCleanup
from ..mcp.client.workflow_mcp_unavailable_error import (
    WorkflowMCPUnavailableError,
)
from ..persistence.sqlite.sessions.sqlite_session_factory import (
    SessionFactory,
    close_session,
    no_session,
)
from .agent_tool_task_scope import AgentToolTaskScope
from .ollama_chat_completions_model import metadata_from_model
from .ollama_model_settings import create_ollama_model_settings
from .ollama_watchdog_model import OllamaWatchdogModel

AGENT_NAME = "Local development workflow coordinator"
LOGGER = logging.getLogger(__name__)


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
        """Execute one bounded segment without replaying earlier actions."""
        connected_servers: list[MCPServer] = []
        mcp_servers = self.mcp_server_factory(profile, run, task)
        skill_tools = self.skill_tool_factory(profile, run)
        workspace_tools = self.workspace_tool_factory(profile, run, task)
        session = None if context is None else self.session_factory(context)
        segment_exhausted = False
        result: object
        tool_tasks = AgentToolTaskScope()
        segment_deadline = asyncio.timeout(
            task.watchdogs.segment_timeout_seconds,
        )

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
                model=OllamaWatchdogModel(
                    self.model,
                    task.watchdogs.provider_response_timeout_seconds,
                ),
                tools=[
                    tool_tasks.wrap(tool)
                    for tool in [*skill_tools, *workspace_tools]
                ],
                mcp_servers=list(mcp_servers),
            )
            async with segment_deadline:
                result = await Runner.run(
                    agent,
                    task.prompt,
                    max_turns=profile.run_limits.segment_turns,
                    run_config=RunConfig(
                        tracing_disabled=True,
                        session_input_callback=(
                            _fresh_segment_input
                            if task.continuation_context is not None
                            else None
                        ),
                        model_settings=create_ollama_model_settings(
                            self.settings,
                        ),
                    ),
                    session=session,
                )
        except MaxTurnsExceeded as error:
            result = error.run_data
            segment_exhausted = True
        except APITimeoutError as error:
            raise AgentProviderTimeoutError(
                "Local provider response deadline expired; no retry was made.",
            ) from error
        except TimeoutError as error:
            if not segment_deadline.expired():
                raise
            raise AgentSegmentTimeoutError(
                "SDK segment deadline expired; its actions were not replayed.",
            ) from error
        except APIConnectionError as error:
            message = (
                f"Ollama is unavailable at {self.settings.base_url}. "
                f"Start Ollama and ensure {self.settings.model} is available."
            )
            raise OllamaUnavailableError(message) from error
        finally:
            try:
                await _cleanup_resources(
                    connected_servers,
                    tool_tasks,
                    task,
                    sys.exception(),
                )
            finally:
                close_session(session)

        final_output: object = getattr(result, "final_output", "")
        response = omit_hidden_reasoning(str(final_output))
        input_tokens, output_tokens = _usage_tokens(result)
        return AgentResult(
            response=response,
            segment_exhausted=segment_exhausted,
            turns_used=(
                profile.run_limits.segment_turns
                if segment_exhausted
                else _response_count(result)
            ),
            generation_metadata=metadata_from_model(
                model=self.model,
                model_name=self.settings.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                visible_output=response,
            ),
        )


async def _cleanup_resources(
    servers: Sequence[MCPServer],
    tool_tasks: AgentToolTaskScope,
    task: AgentTask,
    original_error: BaseException | None,
) -> None:
    """Close owned resources within one shared cancellation grace period."""
    cleanup_budget = task.cleanup_budget or AgentCleanupBudget(
        grace_seconds=task.watchdogs.cleanup_grace_seconds,
    )
    cleanup_budget.begin()
    tool_tasks.cancel()
    try:
        async with asyncio.timeout(cleanup_budget.remaining()):
            cleanup_error = await _close_mcp_servers(servers)
            await tool_tasks.close()
            if cleanup_error is not None:
                raise cleanup_error
    except TimeoutError as error:
        _force_owned_mcp_processes(servers)
        LOGGER.warning(
            "Owned resource cleanup deadline expired; MCP process "
            "termination requested. Inspect interrupted diagnostics.",
        )
        if original_error is None:
            raise AgentCleanupTimeoutError(
                "Owned resource cleanup deadline expired; process "
                "termination requested. Inspect authoritative state "
                "before resuming.",
            ) from error
        original_error.add_note("Owned cleanup deadline expired.")
    except asyncio.CancelledError:
        _force_owned_mcp_processes(servers)
        raise
    except Exception:
        _force_owned_mcp_processes(servers)
        if original_error is None:
            raise
        original_error.add_note("Owned resource cleanup failed.")


async def _close_mcp_servers(
    servers: Sequence[MCPServer],
) -> Exception | None:
    """Visit every owned server while retaining the first teardown failure."""
    first_error: Exception | None = None
    for server in reversed(servers):
        try:
            await server.cleanup()
        except Exception as error:
            LOGGER.warning(
                "Owned MCP cleanup failed; remaining owned resources "
                "will still be closed. Inspect interrupted diagnostics.",
            )
            if first_error is None:
                first_error = error
    return first_error


def _force_owned_mcp_processes(servers: Sequence[MCPServer]) -> None:
    """Terminate only child process groups owned by these MCP transports."""
    for server in servers:
        if isinstance(server, MCPProcessCleanup):
            server.force_cleanup()


def _fresh_segment_input(
    _history: list[TResponseInputItem],
    new_items: list[TResponseInputItem],
) -> list[TResponseInputItem]:
    """Retain new input while leaving stored session history untouched."""
    return new_items


def _response_count(result: object) -> int:
    raw_responses = getattr(result, "raw_responses", None)
    if isinstance(raw_responses, list | tuple):
        responses = cast("list[object] | tuple[object, ...]", raw_responses)
        return max(1, len(responses))
    return 1


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
