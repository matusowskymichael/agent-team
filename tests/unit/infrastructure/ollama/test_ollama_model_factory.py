"""Tests for the Ollama model factory."""

import asyncio

import pytest
from agents import OpenAIChatCompletionsModel

from agent_team.infrastructure.ollama.ollama_chat_completions_model import (
    OllamaChatCompletionsModel,
)
from agent_team.infrastructure.ollama.ollama_model_factory import (
    OLLAMA_API_KEY,
    create_ollama_model,
    create_ollama_openai_client,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings


class TestOllamaModelFactory:
    """Ollama model factory behavior tests."""

    @pytest.mark.parametrize("timeout", [900.0, 1800.0])
    def test_provider_response_timeout_is_explicit(
        self,
        timeout: float,
    ) -> None:
        """Use a project deadline rather than the SDK implicit timeout."""
        settings = OllamaSettings(
            provider_response_timeout_seconds=timeout,
        )
        client = create_ollama_openai_client(settings)
        assert client.timeout == timeout
        assert client.max_retries == 0
        asyncio.run(client.close())

    @pytest.mark.parametrize(
        "timeout", [0.0, -1.0, float("inf"), float("nan")]
    )
    def test_provider_response_timeout_must_be_positive_finite(
        self,
        timeout: float,
    ) -> None:
        """Reject disabled and unbounded local provider deadlines."""
        with pytest.raises(ValueError, match="positive finite"):
            OllamaSettings(provider_response_timeout_seconds=timeout)

    def test_create_ollama_openai_client_uses_local_settings(self) -> None:
        """Configure the OpenAI-compatible client for Ollama."""
        settings = OllamaSettings(
            base_url="http://localhost:4321/v1",
            model="qwen-test",
        )

        client = create_ollama_openai_client(settings)

        assert str(client.base_url) == "http://localhost:4321/v1/"
        assert client.api_key == OLLAMA_API_KEY
        asyncio.run(client.close())

    def test_create_ollama_model_uses_chat_completions_adapter(self) -> None:
        """Use the Agents SDK chat completions model adapter."""
        settings = OllamaSettings(model="qwen-test")

        model = create_ollama_model(settings)

        assert isinstance(model, OpenAIChatCompletionsModel)
        assert isinstance(model, OllamaChatCompletionsModel)
        assert model.model == "qwen-test"
