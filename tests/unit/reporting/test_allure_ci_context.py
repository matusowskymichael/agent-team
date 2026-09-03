"""Tests for safe GitHub Allure context."""

from tests.reporting.allure_ci_context import (
    branch_from,
    github_executor,
    github_links,
)


class TestAllureCIContext:
    """GitHub link and executor validation tests."""

    def test_builds_safe_github_links(self) -> None:
        """Link only validated repository, run, PR, commit, and source data."""
        environment = _github_environment()

        links = github_links(
            "tests/unit/reporting/test_allure_ci_context.py::"
            "TestAllureCIContext::test_builds_safe_github_links",
            environment,
        )

        assert {name for _url, name, _link_type in links} == {
            "Repository",
            "Commit 0123456789ab",
            "Workflow run 456",
            "Pull request #12",
            "Test source",
        }
        assert all("token" not in url for url, _name, _kind in links)

    def test_rejects_untrusted_github_identifiers(self) -> None:
        """Omit links when repository or commit values are unsafe."""
        environment = _github_environment()
        environment["GITHUB_REPOSITORY"] = "owner/repo\nmalicious"
        environment["GITHUB_SHA"] = "not-a-sha"

        assert github_links("tests/unit/test_sample.py", environment) == ()

    def test_builds_valid_executor_metadata(self) -> None:
        """Represent required CI fields through the Allure executor schema."""
        executor = github_executor(_github_environment())

        assert executor is not None
        assert executor["name"] == "GitHub Actions"
        assert executor["type"] == "github"
        assert executor["buildOrder"] == 17
        assert "CI #17" in str(executor["buildName"])
        assert "run 456" in str(executor["buildName"])
        assert "pull/12" in str(executor["buildName"])
        assert executor["reportUrl"] == (
            "https://github.com/matusowskymichael/agent-team/actions/runs/456"
        )

    def test_omits_executor_outside_github_actions(self) -> None:
        """Keep local reports independent of GitHub environment variables."""
        assert github_executor({}) is None
        assert branch_from({}) == "local"


def _github_environment() -> dict[str, str]:
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "matusowskymichael/agent-team",
        "GITHUB_SHA": "0123456789abcdef0123456789abcdef01234567",
        "GITHUB_RUN_ID": "456",
        "GITHUB_RUN_NUMBER": "17",
        "GITHUB_WORKFLOW": "CI",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_REF": "refs/pull/12/merge",
        "GITHUB_REF_NAME": "12/merge",
        "GITHUB_HEAD_REF": "pull/12",
        "IGNORED_TOKEN": "must-not-appear",
    }
