"""Trusted settings and diagnostic callbacks for one candidate attempt."""

from collections.abc import Callable
from dataclasses import dataclass, field

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.runtime.agent_liveness_observer import (
    AgentLivenessObserver,
)
from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)


@dataclass(frozen=True, slots=True)
class EvalCandidateExecutionContext:
    """Supply watchdogs and capture evidence before temporary state is lost."""

    case_timeout_seconds: float = 2700.0
    watchdogs: AgentWatchdogSettings = field(
        default_factory=AgentWatchdogSettings,
    )
    liveness_observer: AgentLivenessObserver | None = None
    checkpoint: Callable[[CandidateRunResult], None] | None = None
