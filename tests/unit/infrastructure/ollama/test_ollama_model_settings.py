"""Tests for Ollama model settings."""

from agent_team.infrastructure.ollama.ollama_model_settings import (
    create_ollama_chat_extra_body,
    create_ollama_model_settings,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings


class TestOllamaModelSettings:
    """Ollama model-settings behavior tests."""

    def test_qwen3_settings_disable_thinking_by_default(self) -> None:
        """Pass local output limit and disabled thinking for Qwen3."""
        settings = OllamaSettings(
            model="qwen3.6:27b",
            max_output_tokens=777,
        )

        model_settings = create_ollama_model_settings(settings)

        assert model_settings.max_tokens == 777
        assert model_settings.include_usage is True
        assert model_settings.preserve_raw_usage is True
        assert model_settings.extra_body == {"think": False}

    def test_qwen3_settings_can_enable_thinking_explicitly(self) -> None:
        """Allow opt-in thinking only for supported Ollama models."""
        settings = OllamaSettings(
            model="qwen3.6:27b",
            thinking_enabled=True,
        )

        assert create_ollama_model_settings(settings).extra_body == {
            "think": True,
        }

    def test_unsupported_models_do_not_receive_thinking_parameter(
        self,
    ) -> None:
        """Avoid sending unsupported thinking fields to other models."""
        settings = OllamaSettings(
            model="llama3.2:3b",
            thinking_enabled=True,
        )

        assert create_ollama_model_settings(settings).extra_body is None
        assert create_ollama_chat_extra_body(settings) is None
