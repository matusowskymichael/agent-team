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

_SENSITIVE_PARAMETER_TERMS = (
    "api_key",
    "authorization",
    "content",
    "credential",
    "password",
    "prompt",
    "secret",
    "token",
)

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
    subsystem = _subsystem(normalized.casefold())
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
        not in {"<redacted>", f"<{type(raw_value).__name__}>"}
        and len(value) <= 48
    ]
    if not readable_parameters:
        return title
    return f"{title} [{', '.join(readable_parameters[:3])}]"


def safe_parameter_value(name: str, value: object) -> str:
    """Return a useful parameter representation that cannot expose secrets."""
    lowered_name = name.casefold()
    if name == "task" or _contains_any(
        lowered_name,
        _SENSITIVE_PARAMETER_TERMS,
    ):
        return "<redacted>"
    if isinstance(value, Enum):
        return str(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        collection = cast("Iterable[object]", value)
        rendered = [_safe_collection_item(item) for item in collection]
        if all(item is not None for item in rendered) and len(rendered) <= 8:
            return "[" + ", ".join(item for item in rendered if item) + "]"
    return f"<{type(cast(object, value)).__name__}>"


def _subsystem(lowered_path: str) -> str:
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
        if term in lowered_path:
            return subsystem
    return "Architecture"


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


def _safe_collection_item(value: object) -> str | None:
    if isinstance(value, Enum):
        return str(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return str(value)
    return None


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)
