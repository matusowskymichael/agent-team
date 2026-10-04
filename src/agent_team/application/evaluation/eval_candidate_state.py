"""Per-attempt diagnostic state shared with trusted runtime observers."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import monotonic

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.runtime.agent_liveness_snapshot import (
    AgentLivenessSnapshot,
)


@dataclass(slots=True)
class EvalCandidateState:
    """Retain bounded liveness and a snapshot before candidate cleanup."""

    checkpoint_callback: Callable[[CandidateRunResult], None] | None = None
    snapshot: AgentLivenessSnapshot | None = None
    partial: CandidateRunResult | None = None
    clock: Callable[[], float] = monotonic
    observed_at: float | None = None

    def observe(self, snapshot: AgentLivenessSnapshot) -> None:
        """Receive trusted diagnostic state without affecting convergence."""
        self.snapshot = snapshot
        self.observed_at = self.clock()

    def current_snapshot(self) -> AgentLivenessSnapshot | None:
        """Advance diagnostic ages while the runtime waits for an operation."""
        if self.snapshot is None or self.observed_at is None:
            return self.snapshot
        elapsed = max(0.0, self.clock() - self.observed_at)
        return replace(
            self.snapshot,
            elapsed_seconds=self.snapshot.elapsed_seconds + elapsed,
            time_since_advancement_seconds=(
                self.snapshot.time_since_advancement_seconds + elapsed
            ),
        )

    def checkpoint(self, candidate: CandidateRunResult) -> None:
        """Persist partial evidence before its temporary workspace closes."""
        prior = self.current_snapshot()
        final = candidate.liveness_snapshot
        if final is not None and prior is not None:
            final = replace(
                final,
                segment_count=max(final.segment_count, prior.segment_count),
                turns_used=max(final.turns_used, prior.turns_used),
                elapsed_seconds=max(
                    final.elapsed_seconds, prior.elapsed_seconds
                ),
                time_since_advancement_seconds=max(
                    final.time_since_advancement_seconds,
                    prior.time_since_advancement_seconds,
                ),
            )
        self.snapshot = final or prior
        self.observed_at = self.clock()
        candidate = replace(candidate, liveness_snapshot=self.snapshot)
        self.partial = candidate
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(candidate)
