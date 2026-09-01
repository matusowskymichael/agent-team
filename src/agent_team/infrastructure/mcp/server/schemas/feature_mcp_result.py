"""MCP result schema for a feature."""

from typing import TypedDict

from agent_team.domain.workflow.feature_status import FeatureStatus


class FeatureMcpResult(TypedDict):
    """Structured MCP representation of a feature."""

    id: int
    title: str
    description: str
    status: FeatureStatus
    created_at: str
    updated_at: str
