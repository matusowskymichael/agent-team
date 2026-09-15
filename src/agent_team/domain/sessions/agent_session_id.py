"""Agent session identity domain model."""

from dataclasses import dataclass

from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class AgentSessionId:
    """Validated persistent local session identity."""

    value: str


def derive_agent_session_id(
    role: DevelopmentRole,
    feature_id: int,
    task_id: int | None = None,
    workspace_identity_hash: str | None = None,
) -> str:
    """Derive a deterministic persisted session ID."""
    if task_id is not None and workspace_identity_hash is not None:
        return (
            f"role-{role.value}-feature-{feature_id}-task-{task_id}-"
            f"workspace-{workspace_identity_hash[:16]}"
        )
    return f"role-{role.value}-feature-{feature_id}"
