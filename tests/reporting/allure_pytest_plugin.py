"""Central pytest hooks for safe and consistent Allure reporting."""

import os
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import allure
import pytest
from allure_commons.types import ParameterMode

from tests.reporting.allure_ci_context import github_links
from tests.reporting.allure_failure_diagnostic import (
    FailureDiagnosticContext,
    failure_diagnostic_json,
)
from tests.reporting.allure_report_metadata import (
    write_allure_report_metadata,
)
from tests.reporting.allure_test_policy import (
    behavior_hierarchy,
    metadata_tags,
    owner_for,
    readable_title,
    safe_parameter_value,
    severity_for,
    stable_test_id,
    suite_hierarchy,
)

# allure-pytest 2.16 exposes these runtime methods without complete parameter
# annotations. Keep that SDK boundary here and type everything else.
_set_parent_suite = cast(
    "Callable[[str], None]",
    allure.dynamic.parent_suite,  # pyright: ignore[reportUnknownMemberType]
)
_set_suite = cast(
    "Callable[[str], None]",
    allure.dynamic.suite,  # pyright: ignore[reportUnknownMemberType]
)
_set_sub_suite = cast(
    "Callable[[str], None]",
    allure.dynamic.sub_suite,  # pyright: ignore[reportUnknownMemberType]
)
_set_epic = cast(
    "Callable[[str], None]",
    allure.dynamic.epic,  # pyright: ignore[reportUnknownMemberType]
)
_set_feature = cast(
    "Callable[[str], None]",
    allure.dynamic.feature,  # pyright: ignore[reportUnknownMemberType]
)
_set_story = cast(
    "Callable[[str], None]",
    allure.dynamic.story,  # pyright: ignore[reportUnknownMemberType]
)
_set_severity = cast(
    "Callable[[str], None]",
    allure.dynamic.severity,  # pyright: ignore[reportUnknownMemberType]
)
_set_label = cast(
    "Callable[[str, str], None]",
    allure.dynamic.label,  # pyright: ignore[reportUnknownMemberType]
)
_set_id = cast(
    "Callable[[str], None]",
    allure.dynamic.id,  # pyright: ignore[reportUnknownMemberType]
)
_set_tag = cast(
    "Callable[[str], None]",
    allure.dynamic.tag,  # pyright: ignore[reportUnknownMemberType]
)
_set_title = cast(
    "Callable[[str], None]",
    allure.dynamic.title,  # pyright: ignore[reportUnknownMemberType]
)


@runtime_checkable
class _ParameterizedCall(Protocol):
    @property
    def params(self) -> Mapping[str, object]:
        """Return pytest's resolved parameter mapping."""
        ...


def pytest_sessionstart(session: pytest.Session) -> None:
    """Write environment, executor, and category metadata when requested."""
    results_directory = _results_directory(session.config)
    if results_directory is None:
        return
    marker_expression = str(session.config.getoption("markexpr", default=""))
    selection = marker_expression or "all configured tests"
    write_allure_report_metadata(
        results_directory,
        test_selection=selection,
        environment=os.environ,
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item) -> None:
    """Apply labels and safe parameters before any fixture setup can fail."""
    if _results_directory(item.config) is None:
        return
    parameters = _parameters(item)
    _set_title(readable_title(item.name, parameters))
    parent_suite, suite, sub_suite = suite_hierarchy(item.nodeid)
    epic, feature, story = behavior_hierarchy(item.nodeid)
    marker_names = tuple(marker.name for marker in item.iter_markers())
    _set_parent_suite(parent_suite)
    _set_suite(suite)
    _set_sub_suite(sub_suite)
    _set_epic(epic)
    _set_feature(feature)
    _set_story(story)
    _set_severity(severity_for(item.nodeid))
    _set_label("owner", owner_for(item.nodeid))
    _set_id(stable_test_id(item.nodeid))
    for tag in metadata_tags(item.nodeid, marker_names):
        _set_tag(tag)
    for url, name, link_type in github_links(item.nodeid, os.environ):
        _set_link(url, link_type, name)
    for name, value in parameters.items():
        display_value = safe_parameter_value(name, value)
        mode = (
            ParameterMode.MASKED
            if display_value == "<redacted>"
            else ParameterMode.DEFAULT
        )
        _set_parameter(name, display_value, None, mode)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(
    item: pytest.Item,
) -> Generator[None, object]:
    """Replace Allure's parameterized name after setup metadata is applied."""
    yield
    if _results_directory(item.config) is not None:
        _set_title(readable_title(item.name, _parameters(item)))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Attach a constrained JSON diagnostic for each failed test phase."""
    report = yield
    if report.failed and _results_directory(item.config) is not None:
        exception_type = (
            call.excinfo.type.__name__
            if call.excinfo is not None
            else "Exception"
        )
        diagnostic = failure_diagnostic_json(
            FailureDiagnosticContext(
                node_id=item.nodeid,
                phase=report.when,
                marker_names=tuple(
                    marker.name for marker in item.iter_markers()
                ),
                duration_seconds=report.duration,
                exception_type=exception_type,
            ),
            environment=os.environ,
        )
        allure.attach(
            diagnostic,
            name="Sanitized failure diagnostic",
            attachment_type=allure.attachment_type.JSON,
        )
    return report


def _results_directory(config: pytest.Config) -> Path | None:
    configured = config.getoption("allure_report_dir", default=None)
    if not configured:
        return None
    return Path(str(configured))


def _set_link(url: str, link_type: str, name: str) -> None:
    setter = cast(
        "Callable[[str, str, str], None]",
        allure.dynamic.link,  # pyright: ignore[reportUnknownMemberType]
    )
    setter(url, link_type, name)


def _set_parameter(
    name: str,
    value: object,
    excluded: bool | None,
    mode: ParameterMode,
) -> None:
    setter = cast(
        "Callable[[str, object, bool | None, ParameterMode], None]",
        allure.dynamic.parameter,  # pyright: ignore[reportUnknownMemberType]
    )
    setter(name, value, excluded, mode)


def _parameters(item: pytest.Item) -> Mapping[str, object]:
    candidate = cast(object, getattr(item, "callspec", None))
    if isinstance(candidate, _ParameterizedCall):
        return candidate.params
    return {}
