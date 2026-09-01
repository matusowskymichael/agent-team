"""Tests for the Ollama model factory."""

import asyncio

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
