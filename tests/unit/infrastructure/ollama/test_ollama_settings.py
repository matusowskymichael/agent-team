"""Tests for Ollama settings."""

import pytest

from agent_team.infrastructure.ollama.ollama_settings import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_BASE_URL_ENV,
    OLLAMA_MAX_OUTPUT_TOKENS_ENV,
    OLLAMA_MODEL_ENV,
    OLLAMA_THINKING_ENABLED_ENV,
    load_ollama_settings,
)


class TestOllamaSettings:
    """Ollama settings behavior tests."""

    def test_load_ollama_settings_uses_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use local Ollama defaults when environment variables are absent."""
        monkeypatch.delenv(OLLAMA_BASE_URL_ENV, raising=False)
        monkeypatch.delenv(OLLAMA_MAX_OUTPUT_TOKENS_ENV, raising=False)
        monkeypatch.delenv(OLLAMA_MODEL_ENV, raising=False)
        monkeypatch.delenv(OLLAMA_THINKING_ENABLED_ENV, raising=False)

        settings = load_ollama_settings()

        assert settings.base_url == DEFAULT_OLLAMA_BASE_URL
        assert settings.model == DEFAULT_OLLAMA_MODEL
        assert settings.max_output_tokens == DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS
        assert settings.thinking_enabled is False

    def test_load_ollama_settings_uses_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use configured Ollama environment variables when present."""
        monkeypatch.setenv(OLLAMA_BASE_URL_ENV, "http://localhost:1234/v1")
        monkeypatch.setenv(OLLAMA_MODEL_ENV, "qwen-test")
        monkeypatch.setenv(OLLAMA_MAX_OUTPUT_TOKENS_ENV, "1234")
        monkeypatch.setenv(OLLAMA_THINKING_ENABLED_ENV, "true")

        settings = load_ollama_settings()

        assert settings.base_url == "http://localhost:1234/v1"
        assert settings.model == "qwen-test"
        assert settings.max_output_tokens == 1234
        assert settings.thinking_enabled is True

    def test_load_ollama_settings_accepts_explicit_mapping(self) -> None:
        """Use a provided environment mapping for deterministic callers."""
        settings = load_ollama_settings(
            {
                OLLAMA_BASE_URL_ENV: "http://localhost:2345/v1",
                OLLAMA_MODEL_ENV: "qwen-mapping",
            },
        )

        assert settings.base_url == "http://localhost:2345/v1"
        assert settings.model == "qwen-mapping"

    def test_model_override_wins_over_environment(self) -> None:
        """Use the explicit runtime model before OLLAMA_MODEL."""
        settings = load_ollama_settings(
            {OLLAMA_MODEL_ENV: "qwen-env"},
            model_override="qwen-cli",
        )

        assert settings.model == "qwen-cli"

    def test_service_root_strips_openai_compatible_suffix(self) -> None:
        """Expose the Ollama service root beside the OpenAI base URL."""
        settings = load_ollama_settings(
            {OLLAMA_BASE_URL_ENV: "http://localhost:11434/v1/"},
        )

        assert settings.service_root == "http://localhost:11434"

    def test_invalid_output_token_limit_fails(self) -> None:
        """Reject non-positive local generation token limits."""
        with pytest.raises(ValueError, match=OLLAMA_MAX_OUTPUT_TOKENS_ENV):
            load_ollama_settings({OLLAMA_MAX_OUTPUT_TOKENS_ENV: "0"})
