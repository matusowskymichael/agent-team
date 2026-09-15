"""Serializers for workflow MCP results."""

from datetime import UTC, datetime

from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_overview import FeatureOverview
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_contract import (
    TaskVerificationContract,
)

from .schemas.artifact_mcp_result import (
    ArtifactMcpResult,
)
from .schemas.development_task_mcp_result import (
    DevelopmentTaskMcpResult,
)
from .schemas.feature_mcp_result import (
    FeatureMcpResult,
)
from .schemas.feature_overview_mcp_result import (
    FeatureOverviewMcpResult,
)
from .schemas.task_handoff_mcp_result import TaskHandoffMcpResult
from .schemas.task_verification_contract_mcp_result import (
    TaskVerificationContractMcpResult,
)


def serialize_feature(feature: Feature) -> FeatureMcpResult:
    """Serialize a feature for MCP structured content."""
    return {
        "id": feature.id,
        "title": feature.title,
        "description": feature.description,
        "status": feature.status,
        "created_at": _serialize_timestamp(feature.created_at),
        "updated_at": _serialize_timestamp(feature.updated_at),
    }


def serialize_feature_overview(
    overview: FeatureOverview,
) -> FeatureOverviewMcpResult:
    """Serialize a feature overview for MCP structured content."""
    return {
        "feature": serialize_feature(overview.feature),
        "artifacts": [
            serialize_artifact(artifact) for artifact in overview.artifacts
        ],
        "tasks": [serialize_development_task(task) for task in overview.tasks],
    }


def serialize_artifact(artifact: Artifact) -> ArtifactMcpResult:
    """Serialize an artifact for MCP structured content."""
    return {
        "id": artifact.id,
        "feature_id": artifact.feature_id,
        "kind": artifact.kind,
        "content": artifact.content,
        "created_by": artifact.created_by,
        "created_at": _serialize_timestamp(artifact.created_at),
    }


def serialize_development_task(
    task: DevelopmentTask,
) -> DevelopmentTaskMcpResult:
    """Serialize a development task for MCP structured content."""
    return {
        "id": task.id,
        "feature_id": task.feature_id,
        "title": task.title,
        "description": task.description,
        "assigned_role": task.assigned_role,
        "status": task.status,
        "created_at": _serialize_timestamp(task.created_at),
        "updated_at": _serialize_timestamp(task.updated_at),
        "verification_contract": serialize_task_verification_contract(
            task.verification_contract,
        ),
    }


def serialize_task_handoff(handoff: TaskHandoff) -> TaskHandoffMcpResult:
    """Serialize a submitted task handoff for MCP structured content."""
    return {
        "id": handoff.id,
        "task_id": handoff.task_id,
        "agent_run_id": handoff.agent_run_id,
        "submitted_by": handoff.submitted_by,
        "attribution": handoff.attribution,
        "implementation_summary": handoff.implementation_summary,
        "changed_paths": list(handoff.changed_paths),
        "reused_symbols": list(handoff.reused_symbols),
        "new_symbols": list(handoff.new_symbols),
        "reuse_notes": handoff.reuse_notes,
        "checks_attempted": list(handoff.checks_attempted),
        "limitations": handoff.limitations,
        "next_action": handoff.next_action,
        "created_at": _serialize_timestamp(handoff.created_at),
    }


def serialize_task_verification_contract(
    contract: TaskVerificationContract | None,
) -> TaskVerificationContractMcpResult | None:
    """Serialize a task verification contract for MCP structured content."""
    if contract is None:
        return None
    return {
        "profile_name": contract.profile_name,
        "required_checks": list(contract.required_checks),
    }


def _serialize_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()
