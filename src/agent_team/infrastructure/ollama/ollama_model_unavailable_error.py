"""Ollama model availability error."""


class OllamaModelUnavailableError(RuntimeError):
    """Raised when a selected local Ollama model is not installed."""
