"""Real SDK segment boundaries with deterministic offline model responses."""

import asyncio
from dataclasses import replace

import httpx2
import pytest
from agents import UserError
from agents.items import ModelResponse
from agents.usage import Usage
from openai import APIConnectionError
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_session_factory,
)
from tests.integration.ollama.sdk_segment_scenario import SdkSegmentScenario


class TestOllamaSegments:
    """Exercise segment exhaustion and persistence through the SDK Runner."""

    def test_internal_segments_preserve_session_without_replay(
        self,
        sdk_segment_scenario: SdkSegmentScenario,
    ) -> None:
        """Persist tool effects once and omit old history on continuation."""
        scenario = sdk_segment_scenario
        scenario.responses.extend(
            [
                _sdk_tool_response("apply_patch"),
                _sdk_tool_response("run_check"),
                _sdk_final_response(),
            ],
        )

        async def execute_segments() -> None:
            session = scenario.session_factory.create_session(scenario.context)
            await session.add_items(
                [{"role": "assistant", "content": "Historical response."}],
            )
            sqlite_session_factory.close_session(session)
            first = await scenario.executor.execute(
                scenario.task,
                scenario.profile,
                scenario.run,
                scenario.context,
            )
            assert first.segment_exhausted is True
            assert first.turns_used == 2
            assert first.response == ""
            assert first.generation_metadata is not None
            assert first.generation_metadata.input_tokens == 20
            assert first.generation_metadata.output_tokens == 6
            assert scenario.operations == ["apply_patch", "run_check"]
            continued_task = replace(
                scenario.task,
                continuation_context="Patch and check already succeeded.",
            )
            final = await scenario.executor.execute(
                continued_task,
                scenario.profile,
                scenario.run,
                scenario.context,
            )
            assert final.segment_exhausted is False
            assert final.turns_used == 1
            assert final.response == "Finished."
            session = scenario.session_factory.create_session(scenario.context)
            stored = await session.get_items()
            sqlite_session_factory.close_session(session)
            assert stored[0] == {
                "role": "assistant",
                "content": "Historical response.",
            }
            stored_calls = sum(
                item.get("type") == "function_call" for item in stored
            )
            assert stored_calls == 2
            assert len(stored) == 8

        asyncio.run(execute_segments())

        assert scenario.operations == ["apply_patch", "run_check"]
        assert scenario.inputs[0] == [
            {"role": "assistant", "content": "Historical response."},
            {"role": "user", "content": scenario.task.prompt},
        ]
        assert scenario.inputs[-1] == [
            {"role": "user", "content": scenario.task.prompt},
        ]

    @pytest.mark.parametrize("failure", ["provider", "cancel", "runtime"])
    def test_failure_after_mutation_never_replays(
        self,
        sdk_segment_scenario: SdkSegmentScenario,
        failure: str,
    ) -> None:
        """Propagate cancellation and failures without replay."""
        scenario = sdk_segment_scenario
        error: BaseException
        expected: type[BaseException]
        if failure == "provider":
            error = APIConnectionError(
                request=httpx2.Request(
                    "POST",
                    "http://localhost:11434/v1/chat/completions",
                ),
            )
            expected = OllamaUnavailableError
        elif failure == "cancel":
            error = asyncio.CancelledError()
            expected = asyncio.CancelledError
        else:
            error = RuntimeError("Infrastructure failed.")
            expected = RuntimeError
        scenario.responses.extend([_sdk_tool_response("apply_patch"), error])

        with pytest.raises(expected):
            asyncio.run(
                scenario.executor.execute(
                    scenario.task,
                    scenario.profile,
                    scenario.run,
                    scenario.context,
                ),
            )

        assert scenario.operations == ["apply_patch"]
        assert len(scenario.inputs) == 2
        assert scenario.responses == []

    def test_workspace_failure_after_patch_never_replays(
        self,
        sdk_workspace_failure_scenario: SdkSegmentScenario,
    ) -> None:
        """Preserve check failures and cancellation without replaying."""
        scenario = sdk_workspace_failure_scenario
        repository = scenario.repository
        assert repository is not None
        error = scenario.workspace_error
        assert error is not None
        cancelled = isinstance(error, asyncio.CancelledError)
        expected_error = asyncio.CancelledError if cancelled else UserError
        scenario.responses.extend(
            [
                _sdk_tool_response(
                    "apply_patch",
                    '{"path":"backend/auth.py","old_text":"0","new_text":"1"}',
                ),
                _sdk_tool_response("run_check", '{"name":"backend"}'),
                _sdk_final_response(),
            ],
        )
        harness = AgentHarness(
            runtime=scenario.executor,
            audit_repository=scenario.audit,
            workflow_repository=repository,
        )

        with pytest.raises(expected_error) as failure:
            asyncio.run(harness.execute(scenario.task))

        if not cancelled:
            assert failure.value.__cause__ is error
        assert len(scenario.inputs) == 2
        assert len(scenario.responses) == 1
        assert scenario.task.workspace_root is not None
        assert (scenario.task.workspace_root / "backend/auth.py").read_text(
            encoding="utf-8"
        ) == "return 1\n"
        assert scenario.task.task_id is not None
        current = repository.get_task(scenario.task.task_id)
        assert current is not None
        assert current.status is TaskStatus.IN_PROGRESS
        assert repository.latest_task_handoff(current.id) is None
        run = scenario.audit.runs[1]
        assert run.status is AgentRunStatus.FAILED
        assert run.error_type == type(failure.value).__name__
        assert run.termination_reason == (
            "cancelled" if cancelled else "runtime_error"
        )
        assert run.segment_count == 1
        invocations = scenario.audit.list_tool_invocations(run.id)
        assert [invocation.tool_name for invocation in invocations] == [
            "apply_patch",
            "run_check",
        ]
        assert [invocation.status for invocation in invocations] == [
            ToolInvocationStatus.COMPLETED,
            ToolInvocationStatus.ALLOWED
            if cancelled
            else ToolInvocationStatus.FAILED,
        ]
        assert invocations[-1].error_type == (None if cancelled else "OSError")


def _sdk_tool_response(name: str, arguments: str = "{}") -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseFunctionToolCall(
                name=name,
                arguments=arguments,
                call_id=f"call-{name}",
                type="function_call",
            ),
        ],
        usage=Usage(input_tokens=10, output_tokens=3),
        response_id=None,
    )


def _sdk_final_response() -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseOutputMessage(
                id="final-message",
                role="assistant",
                status="completed",
                type="message",
                content=[
                    ResponseOutputText(
                        text="Finished.",
                        type="output_text",
                        annotations=[],
                    ),
                ],
            ),
        ],
        usage=Usage(input_tokens=10, output_tokens=3),
        response_id=None,
    )
