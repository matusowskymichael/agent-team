"""Ollama runtime settings."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS = 8192
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
OLLAMA_MAX_OUTPUT_TOKENS_ENV = "OLLAMA_MAX_OUTPUT_TOKENS"
OLLAMA_THINKING_ENABLED_ENV = "OLLAMA_THINKING_ENABLED"


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    """Settings required to call a local Ollama OpenAI-compatible endpoint."""

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_MODEL
    max_output_tokens: int = DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS
    thinking_enabled: bool = False

    @property
    def service_root(self) -> str:
        """Return the Ollama service root for non-OpenAI local endpoints."""
        clean_base_url = self.base_url.rstrip("/")
        if clean_base_url.endswith("/v1"):
            return clean_base_url.removesuffix("/v1")
        return clean_base_url


def load_ollama_settings(
    environ: Mapping[str, str] | None = None,
    model_override: str | None = None,
) -> OllamaSettings:
    """Load Ollama settings from environment variables."""
    values = os.environ if environ is None else environ
    model = model_override or values.get(OLLAMA_MODEL_ENV)
    return OllamaSettings(
        base_url=values.get(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_BASE_URL),
        model=model or DEFAULT_OLLAMA_MODEL,
        max_output_tokens=_positive_integer(
            values.get(OLLAMA_MAX_OUTPUT_TOKENS_ENV),
            DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS,
        ),
        thinking_enabled=_enabled(values.get(OLLAMA_THINKING_ENABLED_ENV)),
    )


def _positive_integer(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            f"{OLLAMA_MAX_OUTPUT_TOKENS_ENV} must be a positive integer.",
        ) from error
    if parsed < 1:
        raise ValueError(
            f"{OLLAMA_MAX_OUTPUT_TOKENS_ENV} must be a positive integer.",
        )
    return parsed


def _enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
