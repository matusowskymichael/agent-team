"""Ollama chat-completions model with finish-reason capture."""

from typing import TYPE_CHECKING, Literal, overload, override

from agents import OpenAIChatCompletionsModel
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem
from agents.models.interface import ModelTracing
from agents.tool import Tool
from agents.tracing.span_data import GenerationSpanData
from agents.tracing.spans import Span
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from openai.types.responses.response_prompt_param import ResponsePromptParam

from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)

if TYPE_CHECKING:
    from agents.model_settings import ModelSettings


class OllamaChatCompletionsModel(OpenAIChatCompletionsModel):
    """Chat-completions adapter that remembers Ollama finish metadata."""

    def __init__(self, model: str, openai_client: AsyncOpenAI) -> None:
        """Initialize the model adapter with finish-reason tracking."""
        super().__init__(model=model, openai_client=openai_client)
        self._last_finish_reason: str | None = None

    @property
    def last_finish_reason(self) -> str | None:
        """Return the finish reason from the latest chat completion."""
        return self._last_finish_reason

    def capture_finish_reason(self, response: ChatCompletion) -> None:
        """Capture finish metadata from a completed chat response."""
        self._last_finish_reason = _finish_reason(response)

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[True],
        prompt: ResponsePromptParam | None = None,
    ) -> tuple[Response, AsyncStream[ChatCompletionChunk]]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[False] = False,
        prompt: ResponsePromptParam | None = None,
    ) -> ChatCompletion: ...

    @override
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: ResponsePromptParam | None = None,
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        response = await super()._fetch_response(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            span=span,
            tracing=tracing,
            stream=stream,
            prompt=prompt,
        )
        if isinstance(response, ChatCompletion):
            self.capture_finish_reason(response)
        return response


def metadata_from_model(
    model: object,
    model_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    visible_output: str,
) -> AgentGenerationMetadata:
    """Build generation metadata from an Ollama model instance."""
    finish_reason = None
    if isinstance(model, OllamaChatCompletionsModel):
        finish_reason = model.last_finish_reason
    return AgentGenerationMetadata(
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        visible_output_char_count=len(visible_output),
        objectively_truncated=finish_reason == "length",
        model=model_name,
    )


def _finish_reason(response: ChatCompletion) -> str | None:
    if not response.choices:
        return None
    finish_reason = response.choices[0].finish_reason
    return str(finish_reason)
