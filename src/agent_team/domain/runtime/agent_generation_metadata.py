"""Provider-neutral generation metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentGenerationMetadata:
    """Sanitized metadata about one model generation."""

    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    visible_output_char_count: int
    objectively_truncated: bool
    model: str
