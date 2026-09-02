"""Tests for Allure result-directory metadata generation."""

import json
from pathlib import Path
from typing import cast

from tests.reporting.allure_report_metadata import (
    environment_properties,
    write_allure_report_metadata,
)


class TestAllureReportMetadata:
    """Allure environment, executor, and category file tests."""

    def test_writes_local_metadata_without_executor(
        self,
        tmp_path: Path,
    ) -> None:
        """Generate valid local files when GitHub variables are absent."""
        categories_path = _categories_file(tmp_path)
        results_directory = tmp_path / "results"

        written = write_allure_report_metadata(
            results_directory,
            test_selection="not ollama and not ollama_eval",
            environment={"SECRET_TOKEN": "do-not-store"},
            categories_path=categories_path,
        )

        assert {path.name for path in written} == {
            "categories.json",
            "environment.properties",
        }
        properties = (results_directory / "environment.properties").read_text(
            encoding="utf-8",
        )
        assert "ci.provider=local" in properties
        assert "git.branch=local" in properties
        assert "git.commit=unavailable" in properties
        assert "live.ollama.excluded=true" in properties
        assert "do-not-store" not in properties
        assert not (results_directory / "executor.json").exists()

    def test_writes_valid_ci_executor_json(self, tmp_path: Path) -> None:
        """Create executor metadata from the approved GitHub value subset."""
        results_directory = tmp_path / "results"

        write_allure_report_metadata(
            results_directory,
            test_selection="unit",
            environment=_github_environment(),
            categories_path=_categories_file(tmp_path),
        )

        executor = json.loads(
            (results_directory / "executor.json").read_text(
                encoding="utf-8",
            ),
        )
        assert executor["type"] == "github"
        assert executor["buildOrder"] == 3
        assert "private-value" not in json.dumps(executor)

    def test_escapes_properties_and_limits_selection(self) -> None:
        """Prevent control characters from breaking properties syntax."""
        properties = environment_properties(
            "unit=one\nsecret:value",
            {},
        )

        assert "test.selection=unit\\=one?secret\\:value" in properties
        assert "\nsecret" not in properties

    def test_repository_categories_are_valid_and_named(self) -> None:
        """Keep the maintained categories file valid and conservative."""
        categories_path = Path("allure/categories.json")

        categories = cast(
            "list[dict[str, object]]",
            json.loads(categories_path.read_text(encoding="utf-8")),
        )

        assert isinstance(categories, list)
        assert len(categories) == 10
        assert all(category.get("name") for category in categories)
        assert all(category.get("matchedStatuses") for category in categories)


def _categories_file(tmp_path: Path) -> Path:
    path = tmp_path / "categories.json"
    path.write_text(
        '[{"name":"Assertion failure","matchedStatuses":["failed"]}]',
        encoding="utf-8",
    )
    return path


def _github_environment() -> dict[str, str]:
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "owner/project",
        "GITHUB_SHA": "fedcba9876543210fedcba9876543210fedcba98",
        "GITHUB_RUN_ID": "99",
        "GITHUB_RUN_NUMBER": "3",
        "GITHUB_WORKFLOW": "CI",
        "GITHUB_REF_NAME": "main",
        "SECRET_TOKEN": "private-value",
    }
