"""Sanitized metadata for observing an active logical run."""

from dataclasses import dataclass

from agent_team.domain.runtime.agent_lifecycle_phase import AgentLifecyclePhase


@dataclass(frozen=True, slots=True)
class AgentLivenessSnapshot:
    """Bounded facts without prompts, source, hashes or reasoning."""

    segment_count: int = 0
    turns_used: int = 0
    task_status: str | None = None
    lifecycle_phase: AgentLifecyclePhase = AgentLifecyclePhase.STARTED
    last_tool_name: str | None = None
    changed_paths: tuple[str, ...] = ()
    last_check_outcome: str | None = None
    last_verification_outcome: str | None = None
    verification_failure_classification: str | None = None
    time_since_advancement_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    waiting_phase: str = "model"
