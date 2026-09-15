"""Fixtures for isolated Allure pytest subprocess tests."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.reporting.allure_steps import fixture_title

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
@fixture_title("Expose the reporting plugin to an isolated pytest process")
def reporting_plugin_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Make repository test support importable only for the subprocess run."""
    existing_path = os.environ.get("PYTHONPATH")
    paths = [str(_PROJECT_ROOT)]
    if existing_path:
        paths.append(existing_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    yield
