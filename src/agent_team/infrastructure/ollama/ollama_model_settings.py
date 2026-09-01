"""Ollama-specific Agents SDK model settings."""

from agents import ModelSettings

from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings


def create_ollama_model_settings(settings: OllamaSettings) -> ModelSettings:
    """Create Agents SDK model settings for local Ollama chat calls."""
    return ModelSettings(
        max_tokens=settings.max_output_tokens,
        extra_body=_extra_body(settings),
        include_usage=True,
        preserve_raw_usage=True,
    )


def create_ollama_chat_extra_body(
    settings: OllamaSettings,
) -> dict[str, object] | None:
    """Create OpenAI client extra_body values for direct chat calls."""
    return _extra_body(settings)


def _extra_body(settings: OllamaSettings) -> dict[str, object] | None:
    body: dict[str, object] = {}
    if supports_ollama_thinking_parameter(settings.model):
        body["think"] = settings.thinking_enabled
    return body or None


def supports_ollama_thinking_parameter(model: str) -> bool:
    """Return whether the local Ollama model accepts the think parameter."""
    clean_model = model.casefold()
    return clean_model.startswith("qwen3")
