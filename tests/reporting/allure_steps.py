"""Typed wrappers for the incompletely annotated Allure step API."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TypeVar, cast

import allure

_FixtureFunction = TypeVar("_FixtureFunction", bound=Callable[..., object])


def report_step(title: str) -> AbstractContextManager[object]:
    """Return an Allure step context through a typed SDK boundary."""
    factory = cast(
        "Callable[[str], AbstractContextManager[object]]",
        allure.step,  # pyright: ignore[reportUnknownMemberType]
    )
    return factory(title)


def fixture_title(
    title: str,
) -> Callable[[_FixtureFunction], _FixtureFunction]:
    """Return a typed Allure title decorator for a pytest fixture."""
    factory = cast(
        "Callable[[str], Callable[[_FixtureFunction], _FixtureFunction]]",
        allure.title,  # pyright: ignore[reportUnknownMemberType]
    )
    return factory(title)
