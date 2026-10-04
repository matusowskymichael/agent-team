"""Durable lifecycle advancement and semantic failure cycle detection."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_team.application.audit.audit_sanitizer import hash_text
from agent_team.application.runtime.agent_task_snapshot import (
    AgentTaskSnapshot,
)
from agent_team.domain.runtime.agent_lifecycle_phase import AgentLifecyclePhase
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)
from agent_team.domain.workflow.task_status import TaskStatus

PHASE_ORDER = tuple(AgentLifecyclePhase)


@dataclass(slots=True)
class AgentConvergenceState:
    """Separate bounded durable advancement from unbounded novel activity."""

    watchdogs: AgentWatchdogSettings
    clock: Callable[[], float] = time.monotonic
    phase: AgentLifecyclePhase = AgentLifecyclePhase.STARTED
    stagnant_segments: int = 0
    started_at: float = field(default=0.0, init=False)
    last_advancement_at: float = field(default=0.0, init=False)
    advancement_count: int = 0
    _highest_phase: AgentLifecyclePhase = AgentLifecyclePhase.STARTED
    _segment_advanced: bool = False
    _required_checks: frozenset[str] = frozenset()
    _check_outcomes: dict[str, bool] = field(default_factory=dict[str, bool])
    _best_passing_checks: dict[str, frozenset[str]] = field(
        default_factory=dict[str, frozenset[str]],
    )
    _failure_counts: dict[str, int] = field(default_factory=dict[str, int])
    _last_verification_id: int | None = None
    _last_handoff_id: int | None = None

    def __post_init__(self) -> None:
        """Start the injected monotonic advancement watchdog."""
        self.started_at = self.clock()
        self.last_advancement_at = self.started_at

    def seed(self, snapshot: AgentTaskSnapshot | None) -> None:
        """Use preexisting state as a baseline without inventing new work."""
        if snapshot is None:
            return
        contract = snapshot.task.verification_contract
        if contract is not None:
            self._required_checks = frozenset(contract.required_checks)
        self._last_handoff_id = (
            None if snapshot.handoff is None else snapshot.handoff.id
        )
        evidence = snapshot.verification
        if evidence is not None:
            self._last_verification_id = evidence.id
            self._best_passing_checks["verification"] = frozenset(
                check.name
                for check in evidence.checks
                if check.exit_code == 0 and not check.timed_out
            )
            self._check_outcomes = {
                check.name: check.exit_code == 0 and not check.timed_out
                for check in evidence.checks
                if not self._required_checks
                or check.name in self._required_checks
            }
            if self._check_outcomes:
                self._best_passing_checks["check"] = frozenset(
                    name
                    for name, passed in self._check_outcomes.items()
                    if passed
                )
        self.phase = (
            AgentLifecyclePhase.COMPLETED
            if snapshot.task.status is TaskStatus.COMPLETED
            else AgentLifecyclePhase.BLOCKED
            if snapshot.task.status is TaskStatus.BLOCKED
            else AgentLifecyclePhase.VERIFIED
            if evidence is not None
            else AgentLifecyclePhase.SUBMITTED
            if snapshot.handoff is not None
            else AgentLifecyclePhase.ACTIVATED
            if snapshot.task.status is TaskStatus.IN_PROGRESS
            else AgentLifecyclePhase.STARTED
        )
        self._highest_phase = self.phase

    def observe_phase(self, phase: AgentLifecyclePhase) -> None:
        """Advance only to a later milestone, while showing current repair."""
        self.phase = phase
        if PHASE_ORDER.index(phase) > PHASE_ORDER.index(self._highest_phase):
            self._highest_phase = phase
            self._record_advancement()

    def observe_check(
        self, name: str, exit_code: int, timed_out: bool
    ) -> None:
        """Compare required check outcomes independently of patch revisions."""
        if self._required_checks and name not in self._required_checks:
            return
        self.observe_phase(AgentLifecyclePhase.CHECKED)
        self._check_outcomes[name] = exit_code == 0 and not timed_out
        passing = frozenset(
            name for name, passed in self._check_outcomes.items() if passed
        )
        improved = self._observe_passing("check", passing)
        if exit_code != 0 or timed_out:
            self._record_failure(
                "check", (name, exit_code, timed_out), improved
            )

    def invalidate_checks(self) -> None:
        """Require fresh check outcomes after changing implementation."""
        self._check_outcomes.clear()

    def observe_snapshot(self, snapshot: AgentTaskSnapshot) -> None:
        """Observe durable submissions and structured verification effects."""
        if self.phase is AgentLifecyclePhase.STARTED and (
            snapshot.task.status is TaskStatus.IN_PROGRESS
        ):
            self.observe_phase(AgentLifecyclePhase.ACTIVATED)
        if snapshot.handoff is not None and (
            snapshot.handoff.id != self._last_handoff_id
        ):
            self._last_handoff_id = snapshot.handoff.id
            self.observe_phase(AgentLifecyclePhase.SUBMITTED)
        evidence = snapshot.verification
        if evidence is not None and (
            evidence.id != self._last_verification_id
        ):
            self._last_verification_id = evidence.id
            self.observe_phase(AgentLifecyclePhase.VERIFIED)
            passing = frozenset(
                check.name
                for check in evidence.checks
                if check.exit_code == 0 and not check.timed_out
            )
            improved = self._observe_passing("verification", passing)
            if evidence.outcome.value == "failed":
                self._record_failure(
                    "verification", snapshot.verification_state(), improved
                )
        if snapshot.task.status is TaskStatus.COMPLETED:
            self.observe_phase(AgentLifecyclePhase.COMPLETED)
        elif snapshot.task.status is TaskStatus.BLOCKED:
            self.observe_phase(AgentLifecyclePhase.BLOCKED)

    def finish_segment(self) -> None:
        """Count consecutive complete segments without durable advancement."""
        if self._segment_advanced:
            self.stagnant_segments = 0
        else:
            self.stagnant_segments += 1
        self._segment_advanced = False

    def advancement_remaining(self) -> float:
        """Return the safety time left since the last genuine advancement."""
        return self.watchdogs.no_advancement_timeout_seconds - max(
            0.0, self.clock() - self.last_advancement_at
        )

    def require_advancement(self) -> None:
        """Stop equivalent failures, short cycles, stagnation and deadlines."""
        if max(self._failure_counts.values(), default=0) >= (
            self.watchdogs.equivalent_failure_threshold
        ):
            raise AgentStalledError(
                "Agent stalled: repeated equivalent structured failures "
                "show no durable advancement; task state is resumable.",
            )
        if self.stagnant_segments >= self.watchdogs.stagnant_segment_threshold:
            raise AgentStalledError(
                "Agent stalled: "
                f"{self.stagnant_segments} consecutive segments produced "
                "no new successful durable advancement; "
                "task state is resumable.",
            )
        if self.advancement_remaining() <= 0:
            raise AgentStalledError(
                "Agent stalled: the no durable advancement safety deadline "
                "was reached; task state is resumable.",
            )

    def _record_advancement(self) -> None:
        self.advancement_count += 1
        self._segment_advanced = True
        self.last_advancement_at = self.clock()
        self._failure_counts.clear()

    def _observe_passing(self, stage: str, passing: frozenset[str]) -> bool:
        previous = self._best_passing_checks.get(stage)
        if previous is None:
            self._best_passing_checks[stage] = passing
            return False
        if passing > previous:
            self._best_passing_checks[stage] = passing
            self._failure_counts.clear()
            self._record_advancement()
            return True
        return False

    def _record_failure(
        self, stage: str, state: object, improved: bool
    ) -> None:
        if improved:
            return
        key = hash_text(json.dumps((stage, state), sort_keys=True))
        self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
