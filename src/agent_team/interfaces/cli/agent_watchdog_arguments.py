"""Shared CLI options for enabled local execution watchdogs."""

import argparse
import math
from collections.abc import Callable
from typing import cast

from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)


def add_watchdog_arguments(
    parser: argparse.ArgumentParser,
    integer_type: Callable[[str], int],
    *,
    include_stall_segments: bool = True,
) -> None:
    """Advertise finite operation and convergence overrides consistently."""
    defaults = AgentWatchdogSettings()
    for option, timeout in (
        (
            "--provider-timeout-seconds",
            defaults.provider_response_timeout_seconds,
        ),
        ("--segment-timeout-seconds", defaults.segment_timeout_seconds),
        (
            "--advancement-timeout-seconds",
            defaults.no_advancement_timeout_seconds,
        ),
    ):
        parser.add_argument(
            option,
            type=positive_finite_seconds,
            default=timeout,
            help=f"Enabled safety watchdog in seconds; default: {timeout:g}.",
        )
    if include_stall_segments:
        parser.add_argument(
            "--stall-segments",
            type=integer_type,
            default=defaults.stagnant_segment_threshold,
            help="Consecutive stagnant segments; default: 4.",
        )
    parser.add_argument(
        "--equivalent-failure-threshold",
        type=integer_type,
        default=defaults.equivalent_failure_threshold,
        help="Repeated equivalent failure states; default: 3.",
    )
    parser.add_argument(
        "--cleanup-grace-seconds",
        type=_cleanup_grace_seconds,
        default=defaults.cleanup_grace_seconds,
        help="Cancellation cleanup grace, at most 10 seconds; default: 10.",
    )


def watchdogs_from_arguments(
    arguments: argparse.Namespace,
) -> AgentWatchdogSettings:
    """Build immutable watchdog policy from validated operator arguments."""
    threshold = cast("int | None", arguments.stall_segments)
    return AgentWatchdogSettings(
        provider_response_timeout_seconds=cast(
            "float", arguments.provider_timeout_seconds
        ),
        segment_timeout_seconds=cast(
            "float", arguments.segment_timeout_seconds
        ),
        no_advancement_timeout_seconds=cast(
            "float", arguments.advancement_timeout_seconds
        ),
        stagnant_segment_threshold=(
            AgentWatchdogSettings().stagnant_segment_threshold
            if threshold is None
            else threshold
        ),
        equivalent_failure_threshold=cast(
            "int", arguments.equivalent_failure_threshold
        ),
        cleanup_grace_seconds=cast("float", arguments.cleanup_grace_seconds),
    )


def positive_finite_seconds(value: str) -> float:
    """Reject disabled, unbounded, and malformed watchdog durations."""
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be positive finite seconds"
        ) from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be positive finite seconds")
    return seconds


def format_watchdog_settings(
    watchdogs: AgentWatchdogSettings, case_timeout_seconds: float
) -> str:
    """Render all enabled evaluation safety bounds for the operator."""
    return (
        "Watchdogs: "
        f"provider={watchdogs.provider_response_timeout_seconds:g}s | "
        f"segment={watchdogs.segment_timeout_seconds:g}s | "
        f"advancement={watchdogs.no_advancement_timeout_seconds:g}s | "
        f"case={case_timeout_seconds:g}s | "
        f"stagnant_segments={watchdogs.stagnant_segment_threshold} | "
        f"equivalent_failures={watchdogs.equivalent_failure_threshold} | "
        f"cleanup={watchdogs.cleanup_grace_seconds:g}s"
    )


def _cleanup_grace_seconds(value: str) -> float:
    seconds = positive_finite_seconds(value)
    if seconds > AgentWatchdogSettings().cleanup_grace_seconds:
        raise argparse.ArgumentTypeError("must be at most 10 seconds")
    return seconds
