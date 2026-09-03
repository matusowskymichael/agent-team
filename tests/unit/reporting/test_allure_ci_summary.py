"""Tests for the concise Allure GitHub summary."""

import json
from pathlib import Path

from tests.reporting.allure_ci_summary import (
    github_summary,
    result_counts,
)


class TestAllureCISummary:
    """Raw result counting and safe Markdown summary tests."""

    def test_counts_result_statuses_and_invalid_results(
        self,
        tmp_path: Path,
    ) -> None:
        """Count statuses without reading names or attachment contents."""
        (tmp_path / "one-result.json").write_text(
            json.dumps({"name": "private", "status": "passed"}),
            encoding="utf-8",
        )
        (tmp_path / "two-result.json").write_text(
            "not-json",
            encoding="utf-8",
        )

        counts = result_counts(tmp_path)

        assert counts == {
            "passed": 1,
            "failed": 0,
            "broken": 0,
            "skipped": 0,
            "unknown": 1,
        }

    def test_renders_statuses_and_artifact_names(self) -> None:
        """Show only aggregate status and expected artifact names."""
        summary = github_summary(
            {
                "passed": 2,
                "failed": 1,
                "broken": 0,
                "skipped": 1,
                "unknown": 0,
            },
            pytest_status="failure",
            report_status="success",
            agent_status="success",
            quality_gate_status="failure",
        )

        assert "Tests: **4** total" in summary
        assert "Quality gate: `failure`" in summary
        assert "`allure-results`" in summary
        assert "`allure-report`" in summary
        assert "`allure-agent-report`" in summary
