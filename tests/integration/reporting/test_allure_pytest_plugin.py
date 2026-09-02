"""Integration tests for the repository's centralized pytest plugin."""

import json
from pathlib import Path
from typing import cast

import pytest


class TestAllurePytestPlugin:
    """Nested pytest execution tests for opt-in Allure output."""

    def test_plain_pytest_run_creates_no_allure_directory(
        self,
        pytester: pytest.Pytester,
    ) -> None:
        """Keep ordinary pytest independent of report generation."""
        pytester.makepyfile(
            test_sample="""
            class TestSample:
                def test_application_behavior(self):
                    assert 2 + 2 == 4
            """,
        )

        result = pytester.runpytest_subprocess(
            "-o",
            "addopts=",
            "-p",
            "tests.reporting.allure_pytest_plugin",
        )

        result.assert_outcomes(passed=1)
        assert not (pytester.path / "allure-results").exists()

    def test_allure_run_records_hierarchy_and_safe_parameters(
        self,
        pytester: pytest.Pytester,
    ) -> None:
        """Emit centralized metadata without changing test behavior."""
        pytester.makepyfile(
            test_sample="""
            import pytest

            class TestSample:
                @pytest.mark.parametrize(
                    ("role", "prompt"),
                    [("business_analyst", "private prompt")],
                )
                def test_application_behavior(self, role, prompt):
                    assert role == "business_analyst"
                    assert prompt == "private prompt"
            """,
        )

        result = pytester.runpytest_subprocess(
            "-o",
            "addopts=",
            "-p",
            "tests.reporting.allure_pytest_plugin",
            "--alluredir=allure-results",
            "--clean-alluredir",
        )

        result.assert_outcomes(passed=1)
        result_data = _single_result(pytester.path / "allure-results")
        result_labels = cast(
            "list[dict[str, str]]",
            result_data["labels"],
        )
        labels = {(label["name"], label["value"]) for label in result_labels}
        result_parameters = cast(
            "list[dict[str, str]]",
            result_data["parameters"],
        )
        parameters = {
            parameter["name"]: parameter["value"]
            for parameter in result_parameters
        }
        assert ("parentSuite", "Unit") in labels
        assert ("epic", "Agent Team") in labels
        assert ("owner", "architecture") in labels
        assert any(name == "as_id" for name, _value in labels)
        assert parameters["role"] == "'business_analyst'"
        assert "private prompt" not in parameters["prompt"]
        assert (pytester.path / "allure-results/categories.json").exists()
        assert (
            pytester.path / "allure-results/environment.properties"
        ).exists()

    def test_failed_run_attaches_sanitized_diagnostic(
        self,
        pytester: pytest.Pytester,
    ) -> None:
        """Attach safe phase metadata while preserving pytest failure."""
        pytester.makepyfile(
            test_sample="""
            class TestSample:
                def test_failure(self):
                    assert False, "ordinary failure"
            """,
        )

        result = pytester.runpytest_subprocess(
            "-o",
            "addopts=",
            "-p",
            "tests.reporting.allure_pytest_plugin",
            "--alluredir=allure-results",
            "--clean-alluredir",
        )

        result.assert_outcomes(failed=1)
        results_directory = pytester.path / "allure-results"
        result_data = _single_result(results_directory)
        attachments = cast(
            "list[dict[str, str]]",
            result_data["attachments"],
        )
        attachment = next(
            attachment
            for attachment in attachments
            if attachment["name"] == "Sanitized failure diagnostic"
        )
        diagnostic = json.loads(
            (results_directory / attachment["source"]).read_text(
                encoding="utf-8",
            ),
        )
        assert diagnostic["phase"] == "call"
        assert diagnostic["exception_type"] == "AssertionError"
        assert "ordinary failure" not in json.dumps(diagnostic)

    def test_setup_failure_keeps_sensitive_parameters_private(
        self,
        pytester: pytest.Pytester,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Redact parameters before a fixture can fail during setup."""
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(autouse=True)
            def fail_during_setup():
                raise RuntimeError("ordinary setup failure")
            """,
        )
        pytester.makepyfile(
            test_sample="""
            import pytest

            class TestSample:
                @pytest.mark.parametrize("prompt", ["private setup prompt"])
                def test_setup_failure(self, prompt):
                    assert prompt
            """,
        )

        result = pytester.runpytest_subprocess(
            "-o",
            "addopts=",
            "-p",
            "tests.reporting.allure_pytest_plugin",
            "--alluredir=allure-results",
            "--clean-alluredir",
        )
        capsys.readouterr()

        result.assert_outcomes(errors=1)
        results_directory = pytester.path / "allure-results"
        result_data = _single_result(results_directory)
        serialized_result = json.dumps(result_data)
        assert "private setup prompt" not in serialized_result
        parameters = cast(
            "list[dict[str, str]]",
            result_data["parameters"],
        )
        assert parameters[0]["mode"] == "masked"
        attachments = cast(
            "list[dict[str, str]]",
            result_data["attachments"],
        )
        attachment = next(
            item
            for item in attachments
            if item["name"] == "Sanitized failure diagnostic"
        )
        diagnostic = json.loads(
            (results_directory / attachment["source"]).read_text(
                encoding="utf-8",
            ),
        )
        assert diagnostic["phase"] == "setup"
        assert diagnostic["exception_type"] == "RuntimeError"


def _single_result(results_directory: Path) -> dict[str, object]:
    result_paths = list(results_directory.glob("*-result.json"))
    assert len(result_paths) == 1
    loaded = cast(
        object,
        json.loads(result_paths[0].read_text(encoding="utf-8")),
    )
    assert isinstance(loaded, dict)
    return cast("dict[str, object]", loaded)
