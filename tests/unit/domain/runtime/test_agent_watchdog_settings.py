"""Safety watchdog defaults and validation."""

from dataclasses import replace

import pytest

from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)


class TestAgentWatchdogSettings:
    """Reject disabled or nonfinite safety settings."""

    def test_watchdog_defaults(self) -> None:
        """Bound unattended execution without imposing a total turn count."""
        watchdogs = AgentWatchdogSettings()
        assert watchdogs.provider_response_timeout_seconds == 900
        assert watchdogs.segment_timeout_seconds == 1800
        assert watchdogs.no_advancement_timeout_seconds == 1800
        assert watchdogs.stagnant_segment_threshold == 4
        assert watchdogs.equivalent_failure_threshold == 3
        assert watchdogs.cleanup_grace_seconds == 10

    @pytest.mark.parametrize(
        "field_name",
        [
            "provider_response_timeout_seconds",
            "segment_timeout_seconds",
            "no_advancement_timeout_seconds",
            "cleanup_grace_seconds",
        ],
    )
    @pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
    def test_watchdogs_reject_invalid_values(
        self, field_name: str, value: float
    ) -> None:
        """All temporal watchdogs must remain positive and finite."""
        with pytest.raises(ValueError, match=field_name):
            replace(AgentWatchdogSettings(), **{field_name: value})

    @pytest.mark.parametrize(
        "field_name",
        ["stagnant_segment_threshold", "equivalent_failure_threshold"],
    )
    @pytest.mark.parametrize("value", [0, -1])
    def test_thresholds_must_be_positive(
        self, field_name: str, value: int
    ) -> None:
        """A disabled convergence detector is not a valid configuration."""
        with pytest.raises(ValueError, match=field_name):
            replace(AgentWatchdogSettings(), **{field_name: value})

    def test_cleanup_grace_is_bounded(self) -> None:
        """Never permit cancellation cleanup to exceed ten seconds."""
        with pytest.raises(ValueError, match="cleanup_grace_seconds"):
            AgentWatchdogSettings(cleanup_grace_seconds=11)
