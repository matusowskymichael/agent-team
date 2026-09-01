"""MCP result schema for a feature overview."""

from typing import TypedDict

from .artifact_mcp_result import (
    ArtifactMcpResult,
)
from .development_task_mcp_result import (
    DevelopmentTaskMcpResult,
)
from .feature_mcp_result import (
    FeatureMcpResult,
)


class FeatureOverviewMcpResult(TypedDict):
    """Structured MCP representation of a feature overview."""

    feature: FeatureMcpResult
    artifacts: list[ArtifactMcpResult]
    tasks: list[DevelopmentTaskMcpResult]
