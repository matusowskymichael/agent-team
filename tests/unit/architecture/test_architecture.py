"""AST-based architecture regression tests."""

import ast
import importlib
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path.cwd()
PRODUCTION_ROOT = PROJECT_ROOT / "src" / "agent_team"
BAD_DUMPING_GROUND_MODULES = {
    "common.py",
    "helpers.py",
    "models.py",
    "ports.py",
    "services.py",
    "types.py",
    "utils.py",
}
FORBIDDEN_LAYER_IMPORTS = {
    "domain": {"application", "infrastructure", "interfaces"},
    "application": {"infrastructure", "interfaces"},
    "infrastructure": {"interfaces"},
}


class TestArchitecture:
    """Project architecture guardrails."""

    def test_production_modules_have_at_most_one_class(self) -> None:
        """Keep one top-level class, enum, dataclass, or protocol per file."""
        violations: list[str] = []
        for path in _production_paths():
            classes = _top_level_classes(path)
            if len(classes) > 1:
                relative_path = path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}: {', '.join(classes)}")

        assert not violations, "\n".join(violations)

    def test_clean_architecture_dependency_direction(self) -> None:
        """Prevent imports from inner layers to outer layers."""
        violations: list[str] = []
        for path in _production_paths():
            importer_layer = _layer_for(_module_name(path))
            if importer_layer is None:
                continue
            forbidden_layers = FORBIDDEN_LAYER_IMPORTS.get(
                importer_layer,
                set(),
            )
            for imported_module in _imported_modules(path):
                imported_layer = _layer_for(imported_module)
                if imported_layer in forbidden_layers:
                    relative_path = path.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"{relative_path} imports {imported_module}",
                    )

        assert not violations, "\n".join(violations)

    def test_init_modules_do_not_create_barrels(self) -> None:
        """Keep package initializers empty except for an optional docstring."""
        violations: list[str] = []
        for path in _production_paths():
            if path.name != "__init__.py":
                continue
            extra_nodes = [
                node
                for node in ast.parse(path.read_text()).body
                if not _is_docstring_node(node)
            ]
            if extra_nodes:
                relative_path = path.relative_to(PROJECT_ROOT)
                violations.append(str(relative_path))

        assert not violations, "\n".join(violations)

    def test_generic_dumping_ground_modules_are_not_introduced(self) -> None:
        """Avoid broad modules such as models.py, services.py, or utils.py."""
        violations = [
            str(path.relative_to(PROJECT_ROOT))
            for path in _production_paths()
            if path.name in BAD_DUMPING_GROUND_MODULES
        ]

        assert not violations, "\n".join(violations)

    def test_configured_entrypoint_modules_import(self) -> None:
        """Ensure configured console-script modules remain importable."""
        pyproject = _mapping(_load_pyproject(), "pyproject")
        project = _mapping(pyproject["project"], "project")
        scripts = _mapping(project["scripts"], "project.scripts")

        failures: list[str] = []
        for name, value in scripts.items():
            if not isinstance(value, str):
                failures.append(f"{name}: entrypoint is not a string")
                continue
            module_name = value.partition(":")[0]
            try:
                importlib.import_module(module_name)
            except Exception as error:
                failures.append(
                    f"{name}: {module_name}: {type(error).__name__}: {error}",
                )

        assert not failures, "\n".join(failures)


def _production_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in PRODUCTION_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
        ),
    )


def _top_level_classes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _imported_modules(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text())
    package = _package_name(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield from _resolve_import_from(node, package)


def _resolve_import_from(
    node: ast.ImportFrom,
    package: str,
) -> Iterator[str]:
    if node.level == 0:
        if node.module is not None:
            yield node.module
        return

    base_module = _resolve_relative_base(package, node.level)
    if base_module is None:
        return
    if node.module is not None:
        yield f"{base_module}.{node.module}"
        return
    for alias in node.names:
        yield f"{base_module}.{alias.name}"


def _resolve_relative_base(package: str, level: int) -> str | None:
    parts = package.split(".")
    if level > len(parts):
        return None
    base_parts = parts[: len(parts) - level + 1]
    return ".".join(base_parts)


def _module_name(path: Path) -> str:
    relative_path = path.relative_to(PROJECT_ROOT / "src")
    return ".".join(relative_path.with_suffix("").parts)


def _package_name(path: Path) -> str:
    module_name = _module_name(path)
    if path.name == "__init__.py":
        return module_name
    return module_name.rpartition(".")[0]


def _layer_for(module_name: str) -> str | None:
    for layer in ("domain", "application", "infrastructure", "interfaces"):
        if module_name == f"agent_team.{layer}":
            return layer
        if module_name.startswith(f"agent_team.{layer}."):
            return layer
    return None


def _is_docstring_node(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _load_pyproject() -> object:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file_handle:
        return tomllib.load(file_handle)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{name} is not a table")
    return cast("Mapping[str, object]", value)
