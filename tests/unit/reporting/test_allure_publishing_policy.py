"""Tests for sanitized Allure publication policy."""

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

    def test_ci_public_pages_deploys_only_sanitized_main_report(
        self,
    ) -> None:
        """Publish only the sanitized report after successful main CI."""
        workflow = Path(".github/workflows/ci.yml").read_text(
            encoding="utf-8",
        )

        upload_step = _step(
            workflow,
            "Upload sanitized Allure report for GitHub Pages",
        )
        pages_job = workflow[workflow.index("\n  publish-pages:") :]

        assert "success()" in upload_step
        assert "github.ref == 'refs/heads/main'" in upload_step
        assert "github.event_name == 'push'" in upload_step
        assert "github.event_name == 'workflow_dispatch'" in upload_step
        assert "path: allure-report" in upload_step
        assert "allure-results-raw" not in upload_step
        assert workflow.index("Enforce checks and reporting") < workflow.index(
            "Upload sanitized Allure report for GitHub Pages",
        )

        assert "needs: test-and-report" in pages_job
        assert "needs.test-and-report.result == 'success'" in pages_job
        assert "github.ref == 'refs/heads/main'" in pages_job
        assert "github.event_name == 'pull_request'" not in pages_job
        assert "pages: write" in pages_job
        assert "id-token: write" in pages_job
        assert "name: github-pages" in pages_job
        assert "actions/configure-pages@" in pages_job
        assert "actions/deploy-pages@" in pages_job
        assert "allure-results-raw" not in pages_job
        assert workflow.count("pages: write") == 1
        assert workflow.count("id-token: write") == 1

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
