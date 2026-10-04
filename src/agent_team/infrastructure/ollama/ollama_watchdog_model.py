"""Provider deadline for every local SDK model request."""

import asyncio
from collections.abc import AsyncGenerator
from typing import override

from agents import Model, ModelSettings, Tool
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseStreamEvent,
)
from agents.models.interface import ModelTracing
from openai import APITimeoutError
from openai.types.responses.response_prompt_param import ResponsePromptParam

from agent_team.domain.runtime.agent_provider_timeout_error import (
    AgentProviderTimeoutError,
)


class OllamaWatchdogModel(Model):
    """Bound a single provider request without replaying inference or tools."""

    def __init__(self, delegate: Model, timeout_seconds: float) -> None:
        """Wrap a run-local model without serializing the SDK client."""
        self.delegate = delegate
        self.timeout_seconds = timeout_seconds

    @override
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        """Cancel a response that exceeds its trusted provider deadline."""
        deadline = asyncio.timeout(self.timeout_seconds)
        try:
            async with deadline:
                return await self.delegate.get_response(
                    system_instructions=system_instructions,
                    input=input,
                    model_settings=model_settings,
                    tools=tools,
                    output_schema=output_schema,
                    handoffs=handoffs,
                    tracing=tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )
        except APITimeoutError as error:
            raise AgentProviderTimeoutError(
                "Local provider response deadline expired; no retry was made.",
            ) from error
        except TimeoutError as error:
            if not deadline.expired():
                raise
            raise AgentProviderTimeoutError(
                "Local provider response deadline expired; no retry was made.",
            ) from error

    @override
    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncGenerator[TResponseStreamEvent]:
        """Bound the stream, including a provider that stops yielding."""
        deadline = asyncio.timeout(self.timeout_seconds)
        try:
            async with deadline:
                async for event in self.delegate.stream_response(
                    system_instructions=system_instructions,
                    input=input,
                    model_settings=model_settings,
                    tools=tools,
                    output_schema=output_schema,
                    handoffs=handoffs,
                    tracing=tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                ):
                    yield event
        except APITimeoutError as error:
            raise AgentProviderTimeoutError(
                "Local provider stream deadline expired; no retry was made.",
            ) from error
        except TimeoutError as error:
            if not deadline.expired():
                raise
            raise AgentProviderTimeoutError(
                "Local provider stream deadline expired; no retry was made.",
            ) from error

    async def _cleanup_on_run_end(self, owner: object) -> None:
        await self.delegate._cleanup_on_run_end(owner)

    @override
    async def close(self) -> None:
        """Release the wrapped SDK model's resources."""
        await self.delegate.close()
