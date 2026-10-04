"""Contained local knowledge package fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def contained_skill_directory(tmp_path: Path) -> Path:
    """Create a script-free skill with one nested Markdown resource."""
    directory = tmp_path / "safe-skill"
    resources = directory / "references"
    resources.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "Reviewed procedure.", encoding="utf-8"
    )
    (resources / "notes.md").write_text("Reviewed notes.", encoding="utf-8")
    return directory
