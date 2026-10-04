"""Regression tests for provider and SDK segment watchdogs."""

import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import httpx2
import pytest
from agents import ModelSettings
from agents.items import ModelResponse, TResponseStreamEvent
from agents.models.interface import ModelTracing
from agents.usage import Usage
from openai import APITimeoutError
from openai.types.responses import ResponseFunctionToolCall

from agent_team.application.evaluation.async_eval_deadline import (
    AsyncEvalDeadline,
)
from agent_team.domain.evaluation.eval_case_timeout_error import (
    EvalCaseTimeoutError,
)
from agent_team.domain.runtime.agent_cleanup_timeout_error import (
    AgentCleanupTimeoutError,
)
from agent_team.domain.runtime.agent_provider_timeout_error import (
    AgentProviderTimeoutError,
)
from agent_team.domain.runtime.agent_segment_timeout_error import (
    AgentSegmentTimeoutError,
)
from agent_team.infrastructure.ollama import ollama_agent_executor
from agent_team.infrastructure.ollama.ollama_watchdog_model import (
    OllamaWatchdogModel,
)
from tests.integration.ollama.sdk_segment_scenario import SdkSegmentScenario
from tests.unit.fakes.mcp.fake_mcp_server import FakeMCPServer
from tests.unit.fakes.runtime.fake_run_result import FakeRunResult


class TestOllamaWatchdogs:
    """Bound hanging requests without retrying an uncertain SDK segment."""

    @pytest.mark.parametrize("outcome", ["normal", "provider", "cancel"])
    def test_cleanup_failure_keeps_primary_termination_and_forces_children(
        self,
        outcome: str,
        monkeypatch: pytest.MonkeyPatch,
        sdk_segment_scenario: SdkSegmentScenario,
    ) -> None:
        """Do not let teardown mask a provider timeout or cancellation."""
        scenario = sdk_segment_scenario
        servers = [FakeMCPServer(tool_names=[]) for _ in range(2)]
        forced: list[bool] = []

        async def cleanup() -> None:
            raise OSError("Owned transport cleanup failed.")

        async def run(*_args: object, **_kwargs: object) -> FakeRunResult:
            if outcome == "provider":
                raise AgentProviderTimeoutError("Provider deadline expired.")
            if outcome == "cancel":
                raise asyncio.CancelledError
            return FakeRunResult(final_output="Completed.")

        for server in servers:
            monkeypatch.setattr(server, "cleanup", cleanup)
            monkeypatch.setattr(
                server,
                "force_cleanup",
                lambda: forced.append(True),
                raising=False,
            )
        monkeypatch.setattr(ollama_agent_executor.Runner, "run", run)

        def server_factory(*_args: object) -> tuple[FakeMCPServer, ...]:
            return tuple(servers)

        executor = replace(
            scenario.executor,
            mcp_server_factory=server_factory,
        )
        expected = {
            "normal": OSError,
            "provider": AgentProviderTimeoutError,
            "cancel": asyncio.CancelledError,
        }[outcome]
        with pytest.raises(expected):
            asyncio.run(
                executor.execute(scenario.task, scenario.profile, scenario.run)
            )
        assert forced == [True, True]

    @pytest.mark.parametrize("outcome", ["normal", "provider", "cancel"])
    def test_cleanup_deadline_forces_owned_termination_and_preserves_error(
        self,
        outcome: str,
        monkeypatch: pytest.MonkeyPatch,
        sdk_segment_scenario: SdkSegmentScenario,
    ) -> None:
        """Force termination while preserving the primary failure."""
        scenario = sdk_segment_scenario
        server = FakeMCPServer(tool_names=[])
        forced: list[bool] = []

        async def cleanup() -> None:
            await asyncio.Event().wait()

        async def run(*_args: object, **_kwargs: object) -> FakeRunResult:
            if outcome == "provider":
                raise AgentProviderTimeoutError("Provider deadline expired.")
            if outcome == "cancel":
                raise asyncio.CancelledError
            return FakeRunResult(final_output="Completed.")

        monkeypatch.setattr(server, "cleanup", cleanup)
        monkeypatch.setattr(
            server, "force_cleanup", lambda: forced.append(True), raising=False
        )
        monkeypatch.setattr(ollama_agent_executor.Runner, "run", run)
        expected = {
            "normal": AgentCleanupTimeoutError,
            "provider": AgentProviderTimeoutError,
            "cancel": asyncio.CancelledError,
        }[outcome]

        def servers(*_args: object) -> tuple[FakeMCPServer, ...]:
            return (server,)

        executor = replace(scenario.executor, mcp_server_factory=servers)
        task = replace(
            scenario.task,
            watchdogs=replace(
                scenario.task.watchdogs,
                cleanup_grace_seconds=0.005,
            ),
        )
        with pytest.raises(expected):
            asyncio.run(executor.execute(task, scenario.profile, scenario.run))
        assert forced == [True]

    @pytest.mark.parametrize("outcome", ["success", "hung", "http", "other"])
    def test_stream_deadline_and_unrelated_timeout_classification(
        self,
        outcome: str,
        monkeypatch: pytest.MonkeyPatch,
        sdk_segment_scenario: SdkSegmentScenario,
    ) -> None:
        """Keep provider stream timeout distinct from unrelated failures."""
        model = sdk_segment_scenario.executor.model
        event = cast("TResponseStreamEvent", {"type": "response.test"})

        async def stream(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[TResponseStreamEvent]:
            yield event
            if outcome == "hung":
                await asyncio.Event().wait()
            elif outcome == "http":
                raise APITimeoutError(
                    request=httpx2.Request("POST", "http://localhost/v1"),
                )
            elif outcome == "other":
                raise TimeoutError("Unrelated adapter failure.")

        monkeypatch.setattr(model, "stream_response", stream)
        wrapper = OllamaWatchdogModel(model, 0.005)

        async def exercise() -> None:
            events = _stream(wrapper)
            assert await anext(events) is event
            if outcome == "success":
                with pytest.raises(StopAsyncIteration):
                    await anext(events)
            else:
                expected = (
                    TimeoutError
                    if outcome == "other"
                    else AgentProviderTimeoutError
                )
                with pytest.raises(expected):
                    await anext(events)
            await events.aclose()

        asyncio.run(exercise())

    @pytest.mark.parametrize("outcome", ["http", "other"])
    def test_response_error_is_not_retried(
        self,
        outcome: str,
        monkeypatch: pytest.MonkeyPatch,
        sdk_segment_scenario: SdkSegmentScenario,
    ) -> None:
        """Translate HTTP timeouts and preserve unrelated adapter failures."""
        model = sdk_segment_scenario.executor.model
        attempts: list[bool] = []

        async def response(*_args: object, **_kwargs: object) -> ModelResponse:
            attempts.append(True)
            if outcome == "http":
                raise APITimeoutError(
                    request=httpx2.Request("POST", "http://localhost/v1"),
                )
            raise TimeoutError("Unrelated adapter failure.")

        monkeypatch.setattr(model, "get_response", response)
        wrapper = OllamaWatchdogModel(model, 1.0)
        expected = (
            AgentProviderTimeoutError if outcome == "http" else TimeoutError
        )
        with pytest.raises(expected):
            asyncio.run(
                wrapper.get_response(
                    None,
                    "test",
                    ModelSettings(),
                    [],
                    None,
                    [],
                    ModelTracing.DISABLED,
                    previous_response_id=None,
                    conversation_id=None,
                    prompt=None,
                )
            )
        assert attempts == [True]

    @pytest.mark.parametrize(
        "termination",
        ["provider", "segment", "case", "cancel"],
    )
    def test_real_mcp_process_and_session_close_on_watchdog(
        self,
        termination: str,
        monkeypatch: pytest.MonkeyPatch,
        sdk_mcp_watchdog_scenario: SdkSegmentScenario,
        tmp_path: Path,
    ) -> None:
        """Timeouts and cancellation leave no owned stdio process or task."""
        scenario = sdk_mcp_watchdog_scenario
        started = asyncio.Event()

        async def hanging(*_args: object, **_kwargs: object) -> object:
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(scenario.executor.model, "get_response", hanging)
        watchdogs = scenario.task.watchdogs
        if termination == "provider":
            watchdogs = replace(
                watchdogs, provider_response_timeout_seconds=0.005
            )
        elif termination == "segment":
            watchdogs = replace(watchdogs, segment_timeout_seconds=0.005)
        task = replace(scenario.task, watchdogs=watchdogs)
        expected = {
            "provider": AgentProviderTimeoutError,
            "segment": AgentSegmentTimeoutError,
            "case": EvalCaseTimeoutError,
            "cancel": asyncio.CancelledError,
        }[termination]

        async def exercise() -> None:
            runtime = scenario.executor.execute(
                task,
                scenario.profile,
                scenario.run,
                scenario.context,
            )
            operation = asyncio.create_task(
                AsyncEvalDeadline().run(runtime, 2.0, 10.0)
                if termination == "case"
                else runtime,
            )
            if termination == "cancel":
                await started.wait()
                operation.cancel()
            with pytest.raises(expected):
                await operation
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(exercise())
        process_id = int((tmp_path / "mcp.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(process_id, 0)

    def test_cancellation_during_real_check_preserves_patch_and_reaps_child(
        self,
        sdk_workspace_cancellation_scenario: SdkSegmentScenario,
        tmp_path: Path,
    ) -> None:
        """Cancel a tool promptly and preserve its preceding patch."""
        scenario = sdk_workspace_cancellation_scenario
        scenario.responses.extend(
            [
                _tool_response(
                    "apply_patch",
                    '{"path":"backend/auth.py","old_text":"0","new_text":"1"}',
                ),
                _tool_response("run_check", '{"name":"backend"}'),
            ]
        )

        async def exercise() -> None:
            operation = asyncio.create_task(
                scenario.executor.execute(
                    scenario.task,
                    scenario.profile,
                    scenario.run,
                )
            )
            async with asyncio.timeout(2):
                while not (tmp_path / "check.pid").exists():
                    await asyncio.sleep(0.001)
                operation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await operation
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(exercise())
        assert (tmp_path / "backend/auth.py").read_text() == "return 1\n"
        process_id = int((tmp_path / "check.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(process_id, 0)

    def test_http_timeout_has_distinct_provider_classification(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sdk_segment_scenario: SdkSegmentScenario,
    ) -> None:
        """Do not classify a provider timeout as connection unavailability."""
        scenario = sdk_segment_scenario

        async def timed_out(*_args: object, **_kwargs: object) -> object:
            raise APITimeoutError(
                request=httpx2.Request("POST", "http://localhost/v1"),
            )

        monkeypatch.setattr(ollama_agent_executor.Runner, "run", timed_out)
        with pytest.raises(AgentProviderTimeoutError):
            asyncio.run(
                scenario.executor.execute(
                    scenario.task,
                    scenario.profile,
                    scenario.run,
                )
            )

    def test_segment_cancels_hung_sdk_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sdk_segment_scenario: SdkSegmentScenario,
    ) -> None:
        """An SDK segment that never returns is cancelled once."""
        scenario = sdk_segment_scenario
        cancelled: list[bool] = []

        async def hanging(*_args: object, **_kwargs: object) -> object:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        monkeypatch.setattr(ollama_agent_executor.Runner, "run", hanging)
        task = replace(
            scenario.task,
            watchdogs=replace(
                scenario.task.watchdogs,
                segment_timeout_seconds=0.005,
            ),
        )

        async def exercise() -> None:
            async with asyncio.timeout(0.5):
                with pytest.raises(AgentSegmentTimeoutError):
                    await scenario.executor.execute(
                        task,
                        scenario.profile,
                        scenario.run,
                    )
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(exercise())
        assert cancelled == [True]

    def test_provider_wrapper_cancels_never_returning_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sdk_segment_scenario: SdkSegmentScenario,
    ) -> None:
        """A hung response has a shorter deadline than its SDK segment."""
        scenario = sdk_segment_scenario
        cancelled: list[bool] = []

        async def hanging(*_args: object, **_kwargs: object) -> object:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        monkeypatch.setattr(scenario.executor.model, "get_response", hanging)
        task = replace(
            scenario.task,
            watchdogs=replace(
                scenario.task.watchdogs,
                provider_response_timeout_seconds=0.005,
            ),
        )

        async def exercise() -> None:
            async with asyncio.timeout(0.5):
                with pytest.raises(AgentProviderTimeoutError):
                    await scenario.executor.execute(
                        task,
                        scenario.profile,
                        scenario.run,
                    )
            assert len(asyncio.all_tasks()) == 1

        asyncio.run(exercise())
        assert cancelled == [True]


def _tool_response(name: str, arguments: str) -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseFunctionToolCall(
                name=name,
                arguments=arguments,
                call_id=f"call-{name}",
                type="function_call",
            )
        ],
        usage=Usage(input_tokens=10, output_tokens=3),
        response_id=None,
    )


def _stream(
    model: OllamaWatchdogModel,
) -> AsyncGenerator[TResponseStreamEvent]:
    return model.stream_response(
        None,
        "test",
        ModelSettings(),
        [],
        None,
        [],
        ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
