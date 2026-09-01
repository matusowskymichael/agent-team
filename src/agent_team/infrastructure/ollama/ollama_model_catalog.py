"""Local Ollama model catalog adapter."""

import json
from collections.abc import Mapping
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from agent_team.infrastructure.ollama.ollama_model_capability_error import (
    OllamaModelCapabilityError,
)
from agent_team.infrastructure.ollama.ollama_model_unavailable_error import (
    OllamaModelUnavailableError,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)

MODEL_CATALOG_TIMEOUT_SECONDS = 5.0


def list_installed_ollama_models(
    settings: OllamaSettings,
    timeout_seconds: float = MODEL_CATALOG_TIMEOUT_SECONDS,
) -> list[str]:
    """Return locally installed Ollama model names."""
    parsed_response = _request_json(
        url=f"{settings.service_root}/api/tags",
        timeout_seconds=timeout_seconds,
    )
    if not isinstance(parsed_response, dict):
        raise OllamaUnavailableError("Ollama returned an invalid model list.")

    response = cast("Mapping[str, object]", parsed_response)
    models_value = response.get("models")
    if not isinstance(models_value, list):
        raise OllamaUnavailableError("Ollama returned an invalid model list.")

    models = cast("list[object]", models_value)
    return sorted(
        name
        for name in (_model_name(model) for model in models)
        if name is not None
    )


def ensure_ollama_model_available(
    settings: OllamaSettings,
    timeout_seconds: float = MODEL_CATALOG_TIMEOUT_SECONDS,
) -> None:
    """Raise a typed error when the configured model is not installed."""
    installed_models = set(
        list_installed_ollama_models(settings, timeout_seconds),
    )
    if settings.model not in installed_models:
        raise OllamaModelUnavailableError(
            f"Ollama model {settings.model!r} is not installed locally.",
        )


def ensure_ollama_model_supports_tools(
    settings: OllamaSettings,
    timeout_seconds: float = MODEL_CATALOG_TIMEOUT_SECONDS,
) -> None:
    """Raise when Ollama reports that the selected model lacks tools."""
    parsed_response = _request_json(
        url=f"{settings.service_root}/api/show",
        timeout_seconds=timeout_seconds,
        body={"model": settings.model},
    )
    if not isinstance(parsed_response, dict):
        return

    response = cast("Mapping[str, object]", parsed_response)
    capabilities_value = response.get("capabilities")
    if not isinstance(capabilities_value, list):
        return

    capabilities = cast("list[object]", capabilities_value)
    capability_names = {
        str(capability)
        for capability in capabilities
        if capability is not None
    }
    if "tools" not in capability_names:
        raise OllamaModelCapabilityError(
            f"Ollama model {settings.model!r} does not report tool support.",
        )


def ensure_ollama_model_ready(
    settings: OllamaSettings,
    timeout_seconds: float = MODEL_CATALOG_TIMEOUT_SECONDS,
) -> None:
    """Validate local installation and tool capability for a runtime model."""
    ensure_ollama_model_available(settings, timeout_seconds)
    ensure_ollama_model_supports_tools(settings, timeout_seconds)


def _model_name(model: object) -> str | None:
    if not isinstance(model, dict):
        return None
    values = cast("Mapping[str, object]", model)
    name = values.get("name")
    if isinstance(name, str) and name:
        return name
    model_id = values.get("model")
    if isinstance(model_id, str) and model_id:
        return model_id
    return None


def _request_json(
    url: str,
    timeout_seconds: float,
    body: dict[str, str] | None = None,
) -> object:
    _require_local_http_url(url)
    request_body = None if body is None else json.dumps(body).encode()
    request = Request(  # noqa: S310 - URL is validated as local HTTP first.
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(  # noqa: S310 - URL is validated as local HTTP first.
            request,
            timeout=timeout_seconds,
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise OllamaUnavailableError(
            "Ollama is unavailable for local model validation.",
        ) from error
    except json.JSONDecodeError as error:
        raise OllamaUnavailableError(
            "Ollama returned invalid JSON during model validation.",
        ) from error


def _require_local_http_url(url: str) -> None:
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise OllamaUnavailableError("Ollama URL must use HTTP.")
    if parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise OllamaUnavailableError(
            "Ollama model validation must target a local service.",
        )
