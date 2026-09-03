"""Derive consistent, safe Allure metadata from pytest identities."""

import re
from collections.abc import Iterable, Mapping
from enum import Enum
from hashlib import sha256
from pathlib import PurePosixPath
from typing import cast

_FEATURES = {
    "Architecture": "Architecture Rules",
    "Audit": "Audit Logging",
    "Context": "Context Construction",
    "Evaluation": "Evaluation Harness",
    "Interfaces": "Command-line Interfaces",
    "MCP": "MCP Integration",
    "Ollama": "Local Ollama Runtime",
    "Persistence": "SQLite Persistence",
    "Reporting": "Test Reporting",
    "Runtime": "Agent Runtime",
    "Sessions": "Feature Sessions",
    "Skills": "Agent Skills",
    "Workflow": "Development Workflow",
    "Workspace": "Workspace Safety",
}

_OWNERS = {
    "Architecture": "architecture",
    "Audit": "observability",
    "Context": "runtime",
    "Evaluation": "evaluation",
    "Interfaces": "interfaces",
    "MCP": "integrations",
    "Ollama": "local-model-runtime",
    "Persistence": "data",
    "Reporting": "test-reporting",
    "Runtime": "runtime",
    "Sessions": "runtime",
    "Skills": "agent-skills",
    "Workflow": "workflow",
    "Workspace": "workspace-security",
}

_REDACTED_PARAMETER = "<redacted>"

_ROLE_PARAMETER_NAMES = {"assigned_role", "role"}
_ROLE_VALUES = {
    "backend_developer",
    "business_analyst",
    "code_reviewer",
    "delivery_manager",
    "frontend_developer",
    "qa_engineer",
    "software_architect",
}
_STATUS_PARAMETER_NAMES = {
    "feature_status",
    "status",
    "task_status",
}
_STATUS_VALUES = {
    "analysis",
    "architecture",
    "blocked",
    "completed",
    "draft",
    "implementation",
    "in_progress",
    "pending",
    "review",
}
_KIND_PARAMETER_NAMES = {"artifact_kind", "kind"}
_KIND_VALUES = {
    "acceptance_criteria",
    "architecture",
    "code_review",
    "implementation_plan",
    "requirements",
    "test_report",
}
_TOOL_PARAMETER_NAMES = {"tool_name"}
_TOOL_COLLECTION_PARAMETER_NAMES = {
    "expected_tool_names",
    "tool_names",
}
_IDENTIFIER_PARAMETER_NAMES = {
    "case_id",
    "field_name",
    "operation",
    "suite_id",
}
_NUMERIC_PARAMETER_NAMES = {
    "attempt",
    "feature_id",
    "limit",
    "max_turns",
    "run_id",
    "task_id",
    "user_version",
}
_BOOLEAN_PARAMETER_NAMES = {
    "enabled",
    "expected",
    "hard_gate_failed",
    "objectively_truncated",
    "reached_mcp",
    "use_structured_content",
}
_EXCEPTION_PARAMETER_NAMES = {
    "error_type",
    "exception_type",
    "expected_error_type",
    "expected_exception",
    "expected_exception_type",
}
_MODEL_PARAMETER_NAMES = {"model"}
_SAFE_IDENTIFIER_VALUE = re.compile(r"[a-z][a-z0-9_.-]{0,79}")
_SAFE_TOOL_VALUE = re.compile(r"[a-z][a-z0-9_]{0,79}")
_SAFE_EXCEPTION_VALUE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.]{0,100}(?:Error|Exception)",
)
_SAFE_MODEL_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,79}")
_ALLOWED_PARAMETER_VALUES = {
    **dict.fromkeys(_ROLE_PARAMETER_NAMES, _ROLE_VALUES),
    **dict.fromkeys(_STATUS_PARAMETER_NAMES, _STATUS_VALUES),
    **dict.fromkeys(_KIND_PARAMETER_NAMES, _KIND_VALUES),
}
_PARAMETER_PATTERNS = {
    **dict.fromkeys(_TOOL_PARAMETER_NAMES, _SAFE_TOOL_VALUE),
    **dict.fromkeys(_IDENTIFIER_PARAMETER_NAMES, _SAFE_IDENTIFIER_VALUE),
    **dict.fromkeys(_EXCEPTION_PARAMETER_NAMES, _SAFE_EXCEPTION_VALUE),
    **dict.fromkeys(_MODEL_PARAMETER_NAMES, _SAFE_MODEL_VALUE),
}

_DIRECTORY_SUBSYSTEMS = {
    "architecture": "Architecture",
    "audit": "Audit",
    "context": "Context",
    "evaluation": "Evaluation",
    "interfaces": "Interfaces",
    "mcp": "MCP",
    "ollama": "Ollama",
    "persistence": "Persistence",
    "reporting": "Reporting",
    "runtime": "Runtime",
    "sessions": "Sessions",
    "skills": "Skills",
    "workflow": "Workflow",
    "workspace": "Workspace",
}

_DISPLAY_WORDS = {
    "api": "API",
    "ci": "CI",
    "cli": "CLI",
    "github": "GitHub",
    "mcp": "MCP",
    "ollama": "Ollama",
    "sdk": "SDK",
    "sqlite": "SQLite",
}

_BLOCKER_TERMS = (
    "authorization",
    "authorized_mcp",
    "capability",
    "cross_feature",
    "feature_binding",
    "foreign_key",
    "no_mutation",
    "rollback",
    "spoof",
    "trusted",
    "unexpected_mutation",
    "workspace_access",
)

_CRITICAL_TERMS = (
    "agent_harness",
    "create_feature",
    "create_task",
    "database_effect",
    "execute",
    "mcp_server",
    "migration",
    "persist",
    "run_agent",
    "schema",
    "update_task",
)


def suite_hierarchy(node_id: str) -> tuple[str, str, str]:
    """Return parent suite, subsystem suite, and component sub-suite."""
    path, *parts = node_id.split("::")
    normalized = path.replace("\\", "/")
    segments = PurePosixPath(normalized).parts
    parent = "Integration" if "integration" in segments else "Unit"
    subsystem = _subsystem(node_id)
    component = _component_name(path, parts)
    return parent, subsystem, component


def behavior_hierarchy(node_id: str) -> tuple[str, str, str]:
    """Return the project epic, feature, and tested story."""
    _parent, subsystem, component = suite_hierarchy(node_id)
    return "Agent Team", _FEATURES[subsystem], component


def metadata_tags(
    node_id: str,
    marker_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return focused tags for the test's layer, subsystem, and risk."""
    parent, subsystem, _component = suite_hierarchy(node_id)
    lowered = node_id.casefold()
    tags = {parent.casefold(), subsystem.casefold()}
    tags.update(marker.casefold() for marker in marker_names)
    if _contains_any(
        lowered,
        (
            "authorization",
            "authorized_mcp",
            "capability",
            "cross_feature",
            "denied",
            "spoof",
            "trusted",
        ),
    ):
        tags.update({"authorization", "security"})
    if "ollama_eval" in tags:
        tags.add("live-evaluation")
    if "mcp" in tags:
        tags.add("workflow")
    return tuple(sorted(tags))


def severity_for(node_id: str) -> str:
    """Assign severity according to the boundary or behavior under test."""
    lowered = node_id.casefold().replace("-", "_")
    if _contains_any(lowered, _BLOCKER_TERMS):
        return "blocker"
    if _contains_any(lowered, _CRITICAL_TERMS):
        return "critical"
    if _contains_any(lowered, ("help", "display", "formatting")):
        return "minor"
    return "normal"


def owner_for(node_id: str) -> str:
    """Return a subsystem responsibility rather than a person."""
    _parent, subsystem, _component = suite_hierarchy(node_id)
    return _OWNERS[subsystem]


def stable_test_id(node_id: str) -> str:
    """Return a deterministic identifier without exposing parameter data."""
    digest = sha256(node_id.encode("utf-8")).hexdigest()[:20]
    return f"pytest-{digest}"


def safe_node_id(node_id: str) -> str:
    """Redact parameter representations while retaining test identity."""
    path = node_id
    parameterized = "[" in node_id and node_id.endswith("]")
    if parameterized:
        path = node_id[: node_id.rfind("[")]
        digest = sha256(node_id.encode("utf-8")).hexdigest()[:12]
        return f"{path}[parameters:{digest}]"
    return path


def readable_title(
    test_name: str,
    parameters: Mapping[str, object],
) -> str:
    """Create a concise title with safe scalar parameter values."""
    base_name = test_name.split("[", maxsplit=1)[0]
    title = _humanize(base_name.removeprefix("test_"))
    readable_parameters = [
        f"{name}={value}"
        for name, raw_value in parameters.items()
        if (value := safe_parameter_value(name, raw_value))
        != _REDACTED_PARAMETER
        and len(value) <= 48
    ]
    if not readable_parameters:
        return title
    return f"{title} [{', '.join(readable_parameters[:3])}]"


def safe_parameter_value(name: str, value: object) -> str:
    """Return an allowlisted safe parameter representation."""
    normalized_name = name.casefold()
    if normalized_name in _TOOL_COLLECTION_PARAMETER_NAMES:
        return _safe_tool_collection(value)
    scalar = _scalar_value(value)
    if scalar is None:
        return _REDACTED_PARAMETER
    is_safe = _is_safe_scalar_parameter(
        normalized_name,
        scalar,
        is_boolean=isinstance(value, bool),
    )
    return scalar if is_safe else _REDACTED_PARAMETER


def safe_reported_parameter_value(name: str, value: object) -> str:
    """Validate an already-rendered Allure parameter against the allowlist."""
    if not isinstance(value, str) or len(value) > 512:
        return _REDACTED_PARAMETER
    normalized_name = name.casefold()
    unquoted = _strip_matching_quotes(value)
    if normalized_name in _TOOL_COLLECTION_PARAMETER_NAMES:
        return _safe_reported_tool_collection(unquoted)
    is_safe = _is_safe_scalar_parameter(
        normalized_name,
        unquoted,
        is_boolean=unquoted in {"True", "False"},
    )
    return value if is_safe else _REDACTED_PARAMETER


def _subsystem(node_id: str) -> str:
    path, *node_parts = node_id.split("::")
    normalized_path = path.replace("\\", "/")
    path_parts = PurePosixPath(normalized_path).parts
    for part in reversed(path_parts[:-1]):
        subsystem = _DIRECTORY_SUBSYSTEMS.get(part.casefold())
        if subsystem is not None:
            return subsystem
    module_subsystem = _subsystem_from_text(
        PurePosixPath(normalized_path).stem,
    )
    if module_subsystem is not None:
        return module_subsystem
    named_subsystem = _subsystem_from_text(" ".join(node_parts))
    return named_subsystem or "Architecture"


def _subsystem_from_text(value: str) -> str | None:
    lowered = value.casefold()
    for term, subsystem in (
        ("evaluation", "Evaluation"),
        ("workspace", "Workspace"),
        ("audit", "Audit"),
        ("sessions", "Sessions"),
        ("mcp", "MCP"),
        ("workflow", "Workflow"),
        ("runtime", "Runtime"),
        ("ollama", "Ollama"),
        ("skills", "Skills"),
        ("context", "Context"),
        ("reporting", "Reporting"),
        ("architecture", "Architecture"),
        ("interfaces", "Interfaces"),
        ("persistence", "Persistence"),
    ):
        if term in lowered:
            return subsystem
    return None


def _component_name(path: str, node_parts: list[str]) -> str:
    if node_parts and node_parts[0].startswith("Test"):
        return _humanize(node_parts[0].removeprefix("Test"))
    stem = PurePosixPath(path.replace("\\", "/")).stem
    return _humanize(stem.removeprefix("test_"))


def _humanize(value: str) -> str:
    with_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    with_boundaries = re.sub(
        r"(?<=[A-Z])(?=[A-Z][a-z])",
        "_",
        with_boundaries,
    )
    words = with_boundaries.replace("-", "_").split("_")
    return " ".join(
        _DISPLAY_WORDS.get(word.casefold(), word.title()) for word in words
    )


def _scalar_value(value: object) -> str | None:
    if isinstance(value, type) and issubclass(value, BaseException):
        return value.__name__
    if isinstance(value, Enum):
        enum_value = value.value
        return str(enum_value) if isinstance(enum_value, str | int) else None
    if isinstance(value, str | int | bool):
        return str(value)
    return None


def _safe_tool_collection(value: object) -> str:
    if not isinstance(value, tuple | list | set | frozenset):
        return _REDACTED_PARAMETER
    collection = cast("Iterable[object]", value)
    rendered = [_scalar_value(item) for item in collection]
    if (
        not rendered
        or len(rendered) > 12
        or any(
            item is None or not _SAFE_TOOL_VALUE.fullmatch(item)
            for item in rendered
        )
    ):
        return _REDACTED_PARAMETER
    safe_items = [item for item in rendered if item is not None]
    if isinstance(value, set | frozenset):
        safe_items.sort()
    return "[" + ", ".join(safe_items) + "]"


def _safe_reported_tool_collection(value: str) -> str:
    if not value.startswith("[") or not value.endswith("]"):
        return _REDACTED_PARAMETER
    raw_items = value[1:-1].split(",")
    items = [_strip_matching_quotes(item.strip()) for item in raw_items]
    if (
        not items
        or len(items) > 12
        or any(not _SAFE_TOOL_VALUE.fullmatch(item) for item in items)
    ):
        return _REDACTED_PARAMETER
    return "[" + ", ".join(items) + "]"


def _is_safe_scalar_parameter(
    name: str,
    value: str,
    *,
    is_boolean: bool,
) -> bool:
    allowed_values = _ALLOWED_PARAMETER_VALUES.get(name)
    pattern = _PARAMETER_PATTERNS.get(name)
    return (
        (allowed_values is not None and value in allowed_values)
        or (pattern is not None and pattern.fullmatch(value) is not None)
        or (name in _NUMERIC_PARAMETER_NAMES and value.isdecimal())
        or (name in _BOOLEAN_PARAMETER_NAMES and is_boolean)
    )


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)
