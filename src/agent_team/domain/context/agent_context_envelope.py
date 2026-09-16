"""Agent context envelope domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentContextEnvelope:
    """Feature-scoped context prepared for one agent run."""

    feature_id: int
    session_id: str
    authoritative_context: str
    max_conversation_history_items: int
    task_id: int | None = None
    workspace_identity_hash: str | None = None
