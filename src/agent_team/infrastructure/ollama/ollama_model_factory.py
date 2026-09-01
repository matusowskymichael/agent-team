"""Ollama model factory."""

from openai import AsyncOpenAI

from agent_team.infrastructure.ollama.ollama_chat_completions_model import (
    OllamaChatCompletionsModel,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings

OLLAMA_API_KEY = "ollama"


def create_ollama_openai_client(settings: OllamaSettings) -> AsyncOpenAI:
    """Create an OpenAI-compatible client for a local Ollama endpoint."""
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=OLLAMA_API_KEY,
        max_retries=0,
    )


def create_ollama_model(
    settings: OllamaSettings,
) -> OllamaChatCompletionsModel:
    """Create the Agents SDK chat completions model for Ollama."""
    return OllamaChatCompletionsModel(
        model=settings.model,
        openai_client=create_ollama_openai_client(settings),
    )
