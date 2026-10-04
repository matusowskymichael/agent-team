"""Bounded human-readable formatting of trusted runtime metadata."""

from collections.abc import Collection

from agent_team.domain.runtime.agent_liveness_snapshot import (
    AgentLivenessSnapshot,
)
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workspace.workspace_tool_name import WorkspaceToolName
from agent_team.interfaces.cli.eval_duration_format import format_duration

SAFE_TOOL_NAMES = frozenset(
    {tool.value for tool in WorkflowToolName}
    | {tool.value for tool in WorkspaceToolName}
    | {"load_skill", "load_skill_resource", "list_skills"}
)
SAFE_WAITING_PHASES = frozenset(
    {"model", "tool", "verification", "cleanup", "idle"}
)
SAFE_OUTCOMES = frozenset({"passed", "failed", "timed_out", "blocked"})
FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


def format_liveness_snapshot(
    snapshot: AgentLivenessSnapshot, elapsed_since_snapshot: float = 0.0
) -> str:
    """Print lifecycle metadata without arbitrary strings or private values."""
    advancement_age = snapshot.time_since_advancement_seconds + max(
        0.0, elapsed_since_snapshot
    )
    status = _safe_value(
        snapshot.task_status, {value.value for value in TaskStatus}
    )
    waiting = _safe_value(snapshot.waiting_phase, SAFE_WAITING_PHASES)
    operation = _safe_value(snapshot.last_tool_name, SAFE_TOOL_NAMES)
    check = _safe_value(snapshot.last_check_outcome, SAFE_OUTCOMES)
    verification = _safe_value(
        snapshot.last_verification_outcome, SAFE_OUTCOMES
    )
    classification = _safe_value(
        snapshot.verification_failure_classification,
        {value.value for value in FailureClassification},
    )
    return (
        f"segment {snapshot.segment_count} | turns ~{snapshot.turns_used} | "
        f"task {status} | phase {snapshot.lifecycle_phase.value} | "
        f"waiting {waiting} | last operation {operation} | "
        f"check {check} | verification {verification} ({classification}) | "
        f"last advancement {format_duration(advancement_age)} ago"
    )


def _safe_value(value: str | None, allowed: Collection[str]) -> str:
    return value if value in allowed and value is not None else "-"
