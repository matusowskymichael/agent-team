"""Tests for sanitized Allure failure diagnostics."""

import json

from tests.reporting.allure_failure_diagnostic import (
    FailureDiagnosticContext,
    failure_diagnostic_json,
)


class TestAllureFailureDiagnostic:
    """Failure diagnostic privacy and truncation tests."""

    def test_contains_only_sanitized_failure_context(self) -> None:
        """Exclude parameters, exception messages, and environment secrets."""
        sensitive_value = "never-store-this-value"

        rendered = failure_diagnostic_json(
            FailureDiagnosticContext(
                node_id=(
                    "tests/unit/test_example.py::TestExample::"
                    f"test_failure[prompt={sensitive_value}]"
                ),
                phase="call",
                marker_names=("security",),
                duration_seconds=0.1234567,
                exception_type="AssertionError: never-store-this-value",
            ),
            environment={
                "GITHUB_RUN_ID": "44",
                "GITHUB_SHA": ("0123456789abcdef0123456789abcdef01234567"),
                "SECRET_TOKEN": sensitive_value,
            },
        )

        diagnostic = json.loads(rendered)
        assert diagnostic["phase"] == "call"
        assert diagnostic["exception_type"] == "AssertionError"
        assert diagnostic["github_run_id"] == "44"
        assert diagnostic["duration_seconds"] == 0.123457
        assert sensitive_value not in rendered
        assert "prompt=" not in rendered

    def test_ignores_invalid_ci_values(self) -> None:
        """Do not include unvalidated workflow identifiers."""
        rendered = failure_diagnostic_json(
            FailureDiagnosticContext(
                node_id=(
                    "tests/unit/test_example.py::TestExample::test_failure"
                ),
                phase="setup",
                marker_names=(),
                duration_seconds=-1,
                exception_type="RuntimeError",
            ),
            environment={
                "GITHUB_RUN_ID": "unsafe-id",
                "GITHUB_SHA": "unsafe-sha",
            },
        )

        diagnostic = json.loads(rendered)
        assert "github_run_id" not in diagnostic
        assert "commit_sha" not in diagnostic
        assert diagnostic["duration_seconds"] == 0.0
