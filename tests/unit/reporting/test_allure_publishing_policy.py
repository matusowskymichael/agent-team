"""Tests for private, sanitized Allure publication policy."""

import json
from pathlib import Path
from typing import cast


class TestAllurePublishingPolicy:
    """CI and local command security-boundary tests."""

    def test_ci_publishes_only_successfully_sanitized_results(self) -> None:
        """Gate reports, artifacts, and PR comments on sanitization."""
        workflow = Path(".github/workflows/ci.yml").read_text(
            encoding="utf-8",
        )

        assert "--alluredir=allure-results-raw" in workflow
        assert "--source-dir allure-results-raw" in workflow
        assert "--output-dir allure-results" in workflow
        assert "path: allure-results-raw" not in workflow
        assert "steps.sanitize.outcome == 'success'" in _step(
            workflow,
            "Generate Allure 3 HTML report",
        )
        assert "steps.sanitize.outcome == 'success'" in _step(
            workflow,
            "Generate agent-readable Allure report",
        )
        assert "steps.sanitize.outcome == 'success'" in _step(
            workflow,
            "Upload sanitized Allure results",
        )
        assert "steps.sanitize.outcome == 'success'" in _step(
            workflow,
            "Upload Allure HTML report",
        )
        assert "steps.sanitize.outcome == 'success'" in _step(
            workflow,
            "Upload agent-readable Allure report",
        )
        assert "outputs.sanitization == 'success'" in workflow
        assert "SANITIZE_STATUS" in workflow
        assert 'PYTEST_STATUS: "${{ steps.pytest.outcome }}"' in workflow

    def test_ci_has_no_public_pages_deployment(self) -> None:
        """Keep private-repository reports out of public GitHub Pages."""
        workflow = Path(".github/workflows/ci.yml").read_text(
            encoding="utf-8",
        )

        assert "deploy-pages" not in workflow
        assert "pages: write" not in workflow
        assert "id-token: write" not in workflow
        assert "upload-pages-artifact" not in workflow

    def test_local_report_commands_consume_only_sanitized_results(
        self,
    ) -> None:
        """Keep HTML and agent commands downstream of the safe directory."""
        package = cast(
            "dict[str, object]",
            json.loads(Path("package.json").read_text(encoding="utf-8")),
        )
        scripts = cast("dict[str, str]", package["scripts"])

        assert "allure-results-raw" in scripts["allure:sanitize"]
        assert "--output-dir allure-results" in scripts["allure:sanitize"]
        assert "allure generate allure-results " in scripts["allure:generate"]
        assert (
            "allure agent inspect allure-results " in scripts["allure:agent"]
        )


def _step(workflow: str, name: str) -> str:
    start = workflow.index(f"      - name: {name}")
    remaining = workflow[start + 1 :]
    next_step = remaining.find("\n      - name:")
    return workflow[start:] if next_step == -1 else remaining[:next_step]
