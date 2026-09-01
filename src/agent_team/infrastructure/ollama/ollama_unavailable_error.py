"""Ollama availability error."""


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama endpoint cannot be reached."""
