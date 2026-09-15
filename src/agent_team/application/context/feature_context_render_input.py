"""Immutable input for rendering authoritative feature context."""

from dataclasses import dataclass

from agent_team.domain.context.agent_context_policy import AgentContextPolicy
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_evidence import (
    TaskVerificationEvidence,
)


@dataclass(frozen=True, slots=True)
class FeatureContextRenderInput:
    """Data required to render deterministic feature context."""

    feature: Feature
    artifacts: tuple[Artifact, ...]
    tasks: tuple[DevelopmentTask, ...]
    policy: AgentContextPolicy
    role: DevelopmentRole
    task_id: int | None
    workspace_identity_hash: str | None
    latest_handoff: TaskHandoff | None
    latest_verification: TaskVerificationEvidence | None
