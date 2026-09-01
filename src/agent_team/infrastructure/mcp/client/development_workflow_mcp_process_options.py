"""Process options for the development workflow MCP server."""

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DevelopmentWorkflowMCPProcessOptions:
    """Subprocess configuration for the workflow MCP server."""

    environ: Mapping[str, str] | None = None
    python_executable: str = sys.executable
    cwd: Path | None = None
