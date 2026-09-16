"""Real SDK segment boundaries with deterministic offline model responses."""

import asyncio
from dataclasses import replace

import httpx2
import pytest
from agents.items import ModelResponse
from agents.usage import Usage
from openai import APIConnectionError
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

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


def _sdk_tool_response(name: str) -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseFunctionToolCall(
                name=name,
                arguments="{}",
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
