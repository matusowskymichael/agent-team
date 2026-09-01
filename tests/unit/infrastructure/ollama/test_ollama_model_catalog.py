"""Tests for the local Ollama model catalog."""

import importlib

import pytest

from agent_team.infrastructure.ollama.ollama_model_capability_error import (
    OllamaModelCapabilityError,
)
from agent_team.infrastructure.ollama.ollama_model_catalog import (
    ensure_ollama_model_available,
    ensure_ollama_model_ready,
    list_installed_ollama_models,
)
from agent_team.infrastructure.ollama.ollama_model_unavailable_error import (
    OllamaModelUnavailableError,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings

ollama_model_catalog = importlib.import_module(
    "agent_team.infrastructure.ollama.ollama_model_catalog",
)


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class TestOllamaModelCatalog:
    """Ollama model catalog behavior tests."""

    def test_lists_installed_models(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parse model names from the local Ollama tags endpoint."""

        def urlopen(
            _request: object,
            timeout: float,
        ) -> _Response:
            assert timeout == 5.0
            return _Response(
                b'{"models":[{"name":"qwen3.5:9b"},{"model":"llama"}]}',
            )

        monkeypatch.setattr(ollama_model_catalog, "urlopen", urlopen)

        models = list_installed_ollama_models(OllamaSettings())

        assert models == ["llama", "qwen3.5:9b"]

    def test_unavailable_model_fails_without_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject explicit models that are not locally installed."""

        def list_models(
            _settings: OllamaSettings,
            _timeout_seconds: float,
        ) -> list[str]:
            return ["qwen3.5:9b"]

        monkeypatch.setattr(
            ollama_model_catalog,
            "list_installed_ollama_models",
            list_models,
        )

        with pytest.raises(OllamaModelUnavailableError):
            ensure_ollama_model_available(
                OllamaSettings(model="missing:1b"),
            )

    def test_model_without_tools_fails_with_capability_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject installed models that report no tool capability."""
        responses = iter(
            (
                b'{"models":[{"name":"qwen3.5:9b"}]}',
                b'{"capabilities":["completion"]}',
            ),
        )

        def urlopen(
            _request: object,
            timeout: float,
        ) -> _Response:
            assert timeout == 5.0
            return _Response(next(responses))

        monkeypatch.setattr(ollama_model_catalog, "urlopen", urlopen)

        with pytest.raises(OllamaModelCapabilityError):
            ensure_ollama_model_ready(OllamaSettings(model="qwen3.5:9b"))
