"""Evaluation-only feature context provider."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.application.context.feature_context_builder import (
    DEFAULT_MAX_CONVERSATION_HISTORY_ITEMS,
    FeatureContextBuilder,
)
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.evaluation.eval_context_policy import (
    EvalContextPolicy,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_not_found_error import (
    FeatureNotFoundError,
)
from agent_team.domain.workflow.workflow_repository import WorkflowRepository


@dataclass(frozen=True, slots=True)
class EvaluationContextProvider:
    """Build context for eval cases without changing production context."""

    repository: WorkflowRepository
    context_policy: EvalContextPolicy

    def build_context(
        self,
        feature_id: int,
        role: DevelopmentRole,
        session_id: str,
    ) -> AgentContextEnvelope:
        """Build context according to the evaluation fixture policy."""
        if self.context_policy is EvalContextPolicy.STANDARD_FEATURE_CONTEXT:
            return FeatureContextBuilder(self.repository).build_context(
                feature_id=feature_id,
                role=role,
                session_id=session_id,
            )

        feature = self.repository.get_feature(feature_id)
        if feature is None:
            raise FeatureNotFoundError(f"Feature {feature_id} was not found.")

        if (
            self.context_policy
            is EvalContextPolicy.METADATA_ONLY_FEATURE_CONTEXT
        ):
            context = _metadata_context(feature, role)
        else:
            context = _no_preload_context(feature_id, role)

        return AgentContextEnvelope(
            feature_id=feature_id,
            session_id=session_id,
            authoritative_context=context,
            max_conversation_history_items=(
                DEFAULT_MAX_CONVERSATION_HISTORY_ITEMS
            ),
        )


def _metadata_context(feature: Feature, role: DevelopmentRole) -> str:
    return "\n".join(
        (
            "EVALUATION AUTHORITATIVE CONTEXT",
            "This evaluation case preloads feature metadata only.",
            f"Active role: {role.value}",
            f"Feature ID: {feature.id}",
            f"Feature title: {feature.title}",
            f"Feature description: {feature.description}",
            f"Feature status: {feature.status.value}",
            f"Feature created_at: {_timestamp(feature.created_at)}",
            f"Feature updated_at: {_timestamp(feature.updated_at)}",
            "Artifact contents: not preloaded by evaluation policy.",
            "Development tasks: not preloaded by evaluation policy.",
            "Use workflow tools for artifact, task, or complete-detail facts.",
        ),
    )


def _no_preload_context(feature_id: int, role: DevelopmentRole) -> str:
    return "\n".join(
        (
            "EVALUATION AUTHORITATIVE CONTEXT",
            "This evaluation case preloads no feature workflow data.",
            f"Active role: {role.value}",
            f"Feature scope ID: {feature_id}",
            "Use workflow tools for feature metadata, artifact, or task "
            "facts.",
        ),
    )


def _timestamp(value: datetime) -> str:
    return value.isoformat()
