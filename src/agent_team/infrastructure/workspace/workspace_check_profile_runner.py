"""Trusted aggregate workspace check runner."""

import json
import subprocess
import sys
from pathlib import Path

BACKEND_COMMANDS = (
    ("uv", "run", "ruff", "format", "--check", "."),
    ("uv", "run", "ruff", "check", "."),
    ("uv", "run", "pyright"),
    ("uv", "run", "pytest"),
)
FRONTEND_SCRIPTS = ("lint", "typecheck", "test", "build")


def main(argv: list[str] | None = None) -> int:
    """Run a trusted aggregate check profile."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1 or arguments[0] not in {"backend", "frontend"}:
        print(
            "Usage: workspace-check-profile backend|frontend",
            file=sys.stderr,
        )
        return 2
    commands = (
        BACKEND_COMMANDS
        if arguments[0] == "backend"
        else _frontend_commands(Path.cwd())
    )
    if not commands:
        print("No configured frontend checks were found.", file=sys.stderr)
        return 1
    for command in commands:
        completed = subprocess.run(command, check=False)  # noqa: S603
        if completed.returncode != 0:
            return completed.returncode
    return 0


def _frontend_commands(root: Path) -> tuple[tuple[str, ...], ...]:
    package_json = root / "package.json"
    if not package_json.is_file():
        print("package.json was not found.", file=sys.stderr)
        return ()
    try:
        manifest = json.loads(package_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("package.json is not valid JSON.", file=sys.stderr)
        return ()
    scripts = manifest.get("scripts")
    if not isinstance(scripts, dict):
        print("package.json has no scripts object.", file=sys.stderr)
        return ()
    commands: list[tuple[str, ...]] = []
    for script_name in FRONTEND_SCRIPTS:
        if script_name not in scripts:
            print(
                f"package.json is missing script {script_name}.",
                file=sys.stderr,
            )
            return ()
        commands.append(("npm", "run", script_name))
    return tuple(commands)


if __name__ == "__main__":
    raise SystemExit(main())
