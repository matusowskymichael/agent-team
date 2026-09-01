"""Workflow database path configuration."""

import os
from collections.abc import Mapping
from pathlib import Path

AGENT_TEAM_DB_PATH_ENV = "AGENT_TEAM_DB_PATH"
DEFAULT_WORKFLOW_DB_PATH = Path(".agent_team/workflow.db")


def load_workflow_database_path(
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Load the workflow database path from the environment."""
    values = os.environ if environ is None else environ
    configured_path = values.get(AGENT_TEAM_DB_PATH_ENV)
    if configured_path is None:
        return DEFAULT_WORKFLOW_DB_PATH
    return Path(configured_path)
