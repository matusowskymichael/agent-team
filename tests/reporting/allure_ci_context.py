"""Build safe GitHub links and executor metadata for Allure."""

import re
from collections.abc import Mapping

_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_RUN_ID_PATTERN = re.compile(r"^[0-9]+$")
_REF_PATTERN = re.compile(r"^[A-Za-z0-9_./-]{1,200}$")
_SAFE_LABEL_PATTERN = re.compile(r"[^A-Za-z0-9 ._#:/()@-]")


def github_links(
    node_id: str,
    environment: Mapping[str, str],
) -> tuple[tuple[str, str, str], ...]:
    """Return validated repository, commit, run, PR, and source links."""
    repository = _repository(environment)
    commit_sha = commit_sha_from(environment)
    if repository is None or commit_sha is None:
        return ()
    base_url = f"https://github.com/{repository}"
    links: list[tuple[str, str, str]] = [
        (base_url, "Repository", "repository"),
        (
            f"{base_url}/commit/{commit_sha}",
            f"Commit {commit_sha[:12]}",
            "commit",
        ),
    ]
    run_id = run_id_from(environment)
    if run_id is not None:
        links.append(
            (
                f"{base_url}/actions/runs/{run_id}",
                f"Workflow run {run_id}",
                "workflow",
            ),
        )
    pull_request = _pull_request_number(environment)
    if pull_request is not None:
        links.append(
            (
                f"{base_url}/pull/{pull_request}",
                f"Pull request #{pull_request}",
                "pull_request",
            ),
        )
    source_path = node_id.split("::", maxsplit=1)[0].replace("\\", "/")
    if _safe_source_path(source_path):
        links.append(
            (
                f"{base_url}/blob/{commit_sha}/{source_path}",
                "Test source",
                "source",
            ),
        )
    return tuple(links)


def github_executor(
    environment: Mapping[str, str],
) -> dict[str, str | int] | None:
    """Return a valid Allure executor object for GitHub Actions only."""
    if environment.get("GITHUB_ACTIONS") != "true":
        return None
    repository = _repository(environment)
    run_id = run_id_from(environment)
    run_number = _positive_integer(environment.get("GITHUB_RUN_NUMBER"))
    workflow = _safe_label(environment.get("GITHUB_WORKFLOW"))
    reference = _reference(environment)
    if None in {repository, run_id, run_number, workflow, reference}:
        return None
    assert repository is not None
    assert run_id is not None
    assert run_number is not None
    assert workflow is not None
    assert reference is not None
    repository_url = f"https://github.com/{repository}"
    owner, project = repository.split("/", maxsplit=1)
    return {
        "name": "GitHub Actions",
        "type": "github",
        "buildName": (f"{workflow} #{run_number} (run {run_id}, {reference})"),
        "buildUrl": f"{repository_url}/actions/runs/{run_id}",
        "buildOrder": run_number,
        "reportName": f"{repository} {workflow} #{run_number}",
        "reportUrl": f"https://{owner}.github.io/{project}/",
    }


def commit_sha_from(environment: Mapping[str, str]) -> str | None:
    """Return a full validated GitHub commit SHA."""
    value = environment.get("GITHUB_SHA", "")
    return value.casefold() if _SHA_PATTERN.fullmatch(value) else None


def run_id_from(environment: Mapping[str, str]) -> str | None:
    """Return a validated numeric GitHub workflow run ID."""
    value = environment.get("GITHUB_RUN_ID", "")
    return value if _RUN_ID_PATTERN.fullmatch(value) else None


def branch_from(environment: Mapping[str, str]) -> str:
    """Return a safe branch or pull-request reference."""
    return _reference(environment) or "local"


def _repository(environment: Mapping[str, str]) -> str | None:
    value = environment.get("GITHUB_REPOSITORY", "")
    return value if _REPOSITORY_PATTERN.fullmatch(value) else None


def _reference(environment: Mapping[str, str]) -> str | None:
    value = environment.get("GITHUB_HEAD_REF") or environment.get(
        "GITHUB_REF_NAME",
        "",
    )
    return value if _REF_PATTERN.fullmatch(value) else None


def _pull_request_number(environment: Mapping[str, str]) -> str | None:
    if environment.get("GITHUB_EVENT_NAME") != "pull_request":
        return None
    reference = environment.get("GITHUB_REF", "")
    match = re.fullmatch(r"refs/pull/([0-9]+)/merge", reference)
    return match.group(1) if match else None


def _positive_integer(value: str | None) -> int | None:
    if value is None or not _RUN_ID_PATTERN.fullmatch(value):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _safe_label(value: str | None) -> str | None:
    if not value:
        return None
    sanitized = _SAFE_LABEL_PATTERN.sub("?", value).strip()
    return sanitized[:160] or None


def _safe_source_path(value: str) -> bool:
    return (
        value.startswith("tests/")
        and ".." not in value.split("/")
        and all(character not in value for character in "\r\n?#")
    )
