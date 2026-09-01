"""Centralized audit sanitization helpers."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import cast

from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)

MAX_AUDIT_EXCERPT_LENGTH = 160
REDACTED_VALUE = "[REDACTED]"
SECRET_KEY_TERMS = (
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "credential",
)


def hash_text(value: str) -> str:
    """Return a SHA-256 hash for audit text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_text(value: object) -> str:
    """Return a sanitized, truncated text excerpt."""
    text = _normalize_whitespace(omit_hidden_reasoning(str(value)))
    text = _redact_inline_secrets(text)
    return _truncate(text)


def sanitize_full_text(value: object) -> str:
    """Return sanitized text without audit excerpt truncation."""
    text = omit_hidden_reasoning(str(value))
    return _redact_inline_secrets(text)


def sanitize_error(error: Exception) -> tuple[str, str]:
    """Return a sanitized error type and message."""
    return error.__class__.__name__, sanitize_text(error)


def omit_hidden_reasoning(value: str) -> str:
    """Return text with hidden reasoning blocks replaced."""
    return re.sub(
        r"<think>.*?</think>",
        "[hidden reasoning omitted]",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )


def generation_metadata_to_json(
    metadata: AgentGenerationMetadata | None,
) -> str | None:
    """Return sanitized generation metadata JSON for audit storage."""
    if metadata is None:
        return None
    payload: dict[str, object] = {
        "finish_reason": _optional_sanitized_text(metadata.finish_reason),
        "input_tokens": metadata.input_tokens,
        "output_tokens": metadata.output_tokens,
        "visible_output_char_count": metadata.visible_output_char_count,
        "objectively_truncated": metadata.objectively_truncated,
        "model": sanitize_text(metadata.model),
    }
    return _to_json(payload)


def sanitize_tool_arguments(
    tool_name: str,
    arguments: Mapping[str, object] | None,
) -> tuple[str, str]:
    """Return hash and preview JSON for tool arguments."""
    sanitized = _sanitize_mapping(tool_name, arguments or {})
    sanitized_json = _to_json(sanitized)
    return hash_text(sanitized_json), sanitized_json


def sanitize_tool_result(
    tool_name: str,
    result: object,
) -> tuple[str, str]:
    """Return hash and preview text for a tool result."""
    if isinstance(result, Mapping):
        result_mapping = cast("Mapping[object, object]", result)
        sanitized = _sanitize_mapping(
            tool_name,
            _string_key_mapping(result_mapping),
        )
        sanitized_json = _to_json(sanitized)
        return hash_text(sanitized_json), _truncate(sanitized_json)

    structured_content = getattr(result, "structured_content", None)
    if isinstance(structured_content, Mapping):
        structured_mapping = cast(
            "Mapping[object, object]",
            structured_content,
        )
        sanitized = _sanitize_mapping(
            tool_name,
            _string_key_mapping(structured_mapping),
        )
        sanitized_json = _to_json(sanitized)
        return hash_text(sanitized_json), _truncate(sanitized_json)

    content = getattr(result, "content", None)
    if isinstance(content, Sequence) and not isinstance(content, str):
        content_items = cast("Sequence[object]", content)
        text_parts: list[str] = []
        for item in content_items:
            text_value = getattr(item, "text", None)
            if isinstance(text_value, str):
                text_parts.append(text_value)
        text = " ".join(text_parts)
    else:
        text = str(result)
    preview = sanitize_text(text)
    return hash_text(preview), preview


def _sanitize_mapping(
    tool_name: str,
    values: Mapping[str, object],
) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in values.items():
        if _is_secret_key(key):
            sanitized[key] = REDACTED_VALUE
        elif tool_name in {"search_code", "code_search"} and key == "query":
            query = str(value)
            sanitized["query_hash"] = hash_text(query)
            sanitized["query_length"] = len(query)
        elif tool_name == "create_task" and key == "description":
            description = str(value)
            sanitized["description_hash"] = hash_text(description)
            sanitized["description_length"] = len(description)
        elif tool_name in {"add_artifact", "read_file"} and key == "content":
            content = str(value)
            sanitized["content_hash"] = hash_text(content)
            sanitized["content_length"] = len(content)
        elif tool_name == "apply_patch" and key in {"old_text", "new_text"}:
            text = str(value)
            sanitized[f"{key}_hash"] = hash_text(text)
            sanitized[f"{key}_length"] = len(text)
        elif key == "line_excerpt":
            excerpt = str(value)
            sanitized["line_excerpt_hash"] = hash_text(excerpt)
            sanitized["line_excerpt_length"] = len(excerpt)
        else:
            sanitized[key] = _sanitize_value(tool_name, value)
    return sanitized


def _sanitize_value(tool_name: str, value: object) -> object:
    if isinstance(value, Mapping):
        value_mapping = cast("Mapping[object, object]", value)
        return _sanitize_mapping(
            tool_name,
            _string_key_mapping(value_mapping),
        )
    if isinstance(value, list | tuple):
        values = cast("Sequence[object]", value)
        return [_sanitize_value(tool_name, item) for item in values]
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return sanitize_text(value)


def _to_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _string_key_mapping(values: Mapping[object, object]) -> dict[str, object]:
    return {str(key): value for key, value in values.items()}


def _is_secret_key(key: str) -> bool:
    clean_key = key.lower()
    return any(term in clean_key for term in SECRET_KEY_TERMS)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _redact_inline_secrets(value: str) -> str:
    redacted = value
    for term in SECRET_KEY_TERMS:
        pattern = re.compile(
            rf"({re.escape(term)}\s*[:=]\s*)\S+",
            re.IGNORECASE,
        )
        redacted = pattern.sub(rf"\1{REDACTED_VALUE}", redacted)
    return redacted


def _optional_sanitized_text(value: str | None) -> str | None:
    if value is None:
        return None
    return sanitize_text(value)


def _truncate(value: str) -> str:
    if len(value) <= MAX_AUDIT_EXCERPT_LENGTH:
        return value
    return value[: MAX_AUDIT_EXCERPT_LENGTH - 3] + "..."
