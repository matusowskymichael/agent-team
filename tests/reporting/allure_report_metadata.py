"""Generate safe Allure environment, executor, and category metadata."""

import argparse
import ast
import json
import os
import platform
import re
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from tests.reporting.allure_ci_context import (
    branch_from,
    commit_sha_from,
    github_executor,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATEGORIES_PATH = _PROJECT_ROOT / "allure" / "categories.json"
_SAFE_SELECTION_PATTERN = re.compile(r"[^A-Za-z0-9 _().:=!-]")


def write_allure_report_metadata(
    results_directory: Path,
    *,
    test_selection: str,
    environment: Mapping[str, str],
    categories_path: Path = _DEFAULT_CATEGORIES_PATH,
) -> tuple[Path, ...]:
    """Write whitelisted metadata files to an Allure results directory."""
    results_directory.mkdir(parents=True, exist_ok=True)
    categories = _load_categories(categories_path)
    written_paths = [
        _write_text(
            results_directory / "categories.json",
            json.dumps(categories, indent=2) + "\n",
        ),
        _write_text(
            results_directory / "environment.properties",
            environment_properties(test_selection, environment),
        ),
    ]
    executor_path = results_directory / "executor.json"
    executor = github_executor(environment)
    if executor is None:
        executor_path.unlink(missing_ok=True)
    else:
        written_paths.append(
            _write_text(
                executor_path,
                json.dumps(executor, indent=2, sort_keys=True) + "\n",
            ),
        )
    return tuple(written_paths)


def environment_properties(
    test_selection: str,
    environment: Mapping[str, str],
) -> str:
    """Return escaped Allure environment properties from safe values only."""
    commit_sha = commit_sha_from(environment)
    excluded_markers = _excluded_live_markers(test_selection)
    properties = {
        "python.version": platform.python_version(),
        "operating.system": platform.system(),
        "architecture": platform.machine(),
        "project.version": _project_version(),
        "test.selection": _safe_selection(test_selection),
        "live.ollama.excluded": str(
            "ollama" in excluded_markers,
        ).casefold(),
        "live.ollama_eval.excluded": str(
            "ollama_eval" in excluded_markers,
        ).casefold(),
        "git.branch": branch_from(environment),
        "git.commit": commit_sha[:12] if commit_sha else "unavailable",
        "ci.provider": (
            "GitHub Actions"
            if environment.get("GITHUB_ACTIONS") == "true"
            else "local"
        ),
    }
    return "".join(
        f"{_escape_property(key)}={_escape_property(value)}\n"
        for key, value in properties.items()
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Write metadata for an existing or pending Allure result run."""
    parser = argparse.ArgumentParser(
        description="Generate safe Agent Team Allure metadata.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("allure-results"),
    )
    parser.add_argument(
        "--selection",
        default="all configured tests",
    )
    parsed = parser.parse_args(arguments)
    write_allure_report_metadata(
        parsed.results_dir,
        test_selection=parsed.selection,
        environment=os.environ,
    )
    return 0


def _load_categories(path: Path) -> list[dict[str, object]]:
    loaded = cast(
        object,
        json.loads(path.read_text(encoding="utf-8")),
    )
    if not isinstance(loaded, list) or not loaded:
        message = "Allure categories must be a non-empty JSON list."
        raise ValueError(message)
    categories: list[dict[str, object]] = []
    for raw_entry in cast("list[object]", loaded):
        if not isinstance(raw_entry, dict):
            message = "Each Allure category must be a JSON object."
            raise ValueError(message)
        entry = cast("dict[object, object]", raw_entry)
        if not isinstance(
            entry.get("name"),
            str,
        ):
            message = "Each Allure category must have a string name."
            raise ValueError(message)
        categories.append({str(key): value for key, value in entry.items()})
    return categories


def _write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _project_version() -> str:
    try:
        return version("agent-team")
    except PackageNotFoundError:
        return "unknown"


def _safe_selection(value: str) -> str:
    sanitized = _SAFE_SELECTION_PATTERN.sub("?", value).strip()
    return sanitized[:160] or "all configured tests"


def _excluded_live_markers(value: str) -> frozenset[str]:
    try:
        expression = ast.parse(value.casefold(), mode="eval")
    except SyntaxError:
        return frozenset()
    if not _supported_marker_expression(expression.body):
        return frozenset()
    return frozenset(
        marker
        for marker in ("ollama", "ollama_eval")
        if True
        not in _possible_expression_values(
            expression.body,
            selected_marker=marker,
        )
    )


def _supported_marker_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _supported_marker_expression(node.operand)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And | ast.Or):
        return all(
            _supported_marker_expression(value) for value in node.values
        )
    return False


def _possible_expression_values(
    node: ast.expr,
    *,
    selected_marker: str,
) -> frozenset[bool]:
    if isinstance(node, ast.Name):
        if node.id == selected_marker:
            return frozenset({True})
        return frozenset({False, True})
    if isinstance(node, ast.UnaryOp):
        return frozenset(
            not value
            for value in _possible_expression_values(
                node.operand,
                selected_marker=selected_marker,
            )
        )
    assert isinstance(node, ast.BoolOp)
    possible = _possible_expression_values(
        node.values[0],
        selected_marker=selected_marker,
    )
    for child in node.values[1:]:
        child_values = _possible_expression_values(
            child,
            selected_marker=selected_marker,
        )
        if isinstance(node.op, ast.And):
            possible = frozenset(
                left and right for left in possible for right in child_values
            )
        else:
            possible = frozenset(
                left or right for left in possible for right in child_values
            )
    return possible


def _escape_property(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("=", "\\=")
        .replace(":", "\\:")
    )


if __name__ == "__main__":
    raise SystemExit(main())
