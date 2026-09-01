"""MCP result schema for an artifact."""

from typing import TypedDict

from agent_team.domain.workflow.artifact_kind import ArtifactKind


class ArtifactMcpResult(TypedDict):
    """Structured MCP representation of an artifact."""

    id: int
    feature_id: int
    kind: ArtifactKind
    content: str
    created_by: str
    created_at: str
