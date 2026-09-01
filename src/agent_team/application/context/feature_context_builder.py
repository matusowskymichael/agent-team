"""Application service for authoritative feature context."""

from dataclasses import dataclass, field
from datetime import datetime

from agent_team.domain.context.agent_context_budget_exceeded_error import (
    AgentContextBudgetExceededError,
)
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.context.agent_context_policy import AgentContextPolicy
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_not_found_error import (
    FeatureNotFoundError,
)
from agent_team.domain.workflow.workflow_repository import WorkflowRepository

DEFAULT_MAX_AUTHORITATIVE_CONTEXT_CHARS = 20_000
DEFAULT_MAX_CONVERSATION_HISTORY_ITEMS = 20


def _default_context_policies() -> dict[DevelopmentRole, AgentContextPolicy]:
    return {
        DevelopmentRole.DELIVERY_MANAGER: AgentContextPolicy(
            artifact_kinds=frozenset(ArtifactKind),
            task_roles=frozenset(),
            include_all_tasks=True,
        ),
        DevelopmentRole.BUSINESS_ANALYST: AgentContextPolicy(
            artifact_kinds=frozenset(
                {
                    ArtifactKind.REQUIREMENTS,
                    ArtifactKind.ACCEPTANCE_CRITERIA,
                },
            ),
            task_roles=frozenset(),
        ),
        DevelopmentRole.SOFTWARE_ARCHITECT: AgentContextPolicy(
            artifact_kinds=frozenset(
                {
                    ArtifactKind.REQUIREMENTS,
                    ArtifactKind.ACCEPTANCE_CRITERIA,
                    ArtifactKind.ARCHITECTURE,
                    ArtifactKind.IMPLEMENTATION_PLAN,
                },
            ),
            task_roles=frozenset(),
            include_all_tasks=True,
        ),
        DevelopmentRole.BACKEND_DEVELOPER: AgentContextPolicy(
            artifact_kinds=frozenset(
                {
                    ArtifactKind.REQUIREMENTS,
                    ArtifactKind.ACCEPTANCE_CRITERIA,
                    ArtifactKind.ARCHITECTURE,
                    ArtifactKind.IMPLEMENTATION_PLAN,
                },
            ),
            task_roles=frozenset({DevelopmentRole.BACKEND_DEVELOPER}),
        ),
        DevelopmentRole.FRONTEND_DEVELOPER: AgentContextPolicy(
            artifact_kinds=frozenset(
                {
                    ArtifactKind.REQUIREMENTS,
                    ArtifactKind.ACCEPTANCE_CRITERIA,
                    ArtifactKind.ARCHITECTURE,
                    ArtifactKind.IMPLEMENTATION_PLAN,
                },
            ),
            task_roles=frozenset({DevelopmentRole.FRONTEND_DEVELOPER}),
        ),
        DevelopmentRole.QA_ENGINEER: AgentContextPolicy(
            artifact_kinds=frozenset(
                {
                    ArtifactKind.REQUIREMENTS,
                    ArtifactKind.ACCEPTANCE_CRITERIA,
                    ArtifactKind.ARCHITECTURE,
                    ArtifactKind.IMPLEMENTATION_PLAN,
                    ArtifactKind.TEST_REPORT,
                },
            ),
            task_roles=frozenset(),
            include_all_tasks=True,
        ),
        DevelopmentRole.CODE_REVIEWER: AgentContextPolicy(
            artifact_kinds=frozenset(ArtifactKind),
            task_roles=frozenset(),
            include_all_tasks=True,
        ),
    }


@dataclass(frozen=True, slots=True)
class FeatureContextBuilder:
    """Build deterministic authoritative workflow context for a role."""

    repository: WorkflowRepository
    policies: dict[DevelopmentRole, AgentContextPolicy] = field(
        default_factory=_default_context_policies,
    )

    def build_context(
        self,
        feature_id: int,
        role: DevelopmentRole,
        session_id: str,
    ) -> AgentContextEnvelope:
        """Build fresh least-privilege context for one feature-scoped run."""
        feature = self.repository.get_feature(feature_id)
        if feature is None:
            raise FeatureNotFoundError(f"Feature {feature_id} was not found.")

        policy = self.policies[role]
        artifacts = self._select_artifacts(feature_id, policy)
        tasks = self._select_tasks(feature_id, policy)
        authoritative_context = _render_context(
            feature=feature,
            artifacts=artifacts,
            tasks=tasks,
            policy=policy,
            role=role,
        )
        if len(authoritative_context) > policy.max_authoritative_context_chars:
            raise AgentContextBudgetExceededError(
                "Authoritative workflow context exceeds the configured "
                "character budget.",
            )

        return AgentContextEnvelope(
            feature_id=feature_id,
            session_id=session_id,
            authoritative_context=authoritative_context,
            max_conversation_history_items=(
                policy.max_conversation_history_items
            ),
        )

    def _select_artifacts(
        self,
        feature_id: int,
        policy: AgentContextPolicy,
    ) -> tuple[Artifact, ...]:
        artifacts = self.repository.list_artifacts(feature_id)
        return tuple(
            artifact
            for artifact in sorted(artifacts, key=lambda item: item.id)
            if artifact.kind in policy.artifact_kinds
        )

    def _select_tasks(
        self,
        feature_id: int,
        policy: AgentContextPolicy,
    ) -> tuple[DevelopmentTask, ...]:
        if not policy.include_all_tasks and not policy.task_roles:
            return ()
        tasks = self.repository.list_tasks(feature_id)
        selected_tasks = (
            tasks
            if policy.include_all_tasks
            else [
                task
                for task in tasks
                if task.assigned_role in policy.task_roles
            ]
        )
        return tuple(sorted(selected_tasks, key=lambda item: item.id))


def _render_context(
    feature: Feature,
    artifacts: tuple[Artifact, ...],
    tasks: tuple[DevelopmentTask, ...],
    policy: AgentContextPolicy,
    role: DevelopmentRole,
) -> str:
    lines = [
        "AUTHORITATIVE WORKFLOW CONTEXT",
        "This context is refreshed from the local workflow database for this "
        "run. It outranks conversation history.",
        f"Active role: {role.value}",
        f"Feature ID: {feature.id}",
        f"Feature title: {feature.title}",
        f"Feature description: {feature.description}",
        f"Feature status: {feature.status.value}",
        f"Feature created_at: {_timestamp(feature.created_at)}",
        f"Feature updated_at: {_timestamp(feature.updated_at)}",
        "",
        "Artifacts included by role policy:",
    ]
    _append_artifacts(lines, artifacts)
    lines.extend(
        (
            "",
            "Development tasks included by role policy:",
        ),
    )
    _append_tasks(lines, tasks, policy)
    lines.extend(
        (
            "",
            "Grounding rules:",
            "- Treat workflow artifacts above as authoritative.",
            "- Do not infer absent artifacts or tasks from data not included.",
            "- Use workflow tools to verify current data before making new "
            "factual claims.",
        ),
    )
    return "\n".join(lines)


def _append_artifacts(
    lines: list[str],
    artifacts: tuple[Artifact, ...],
) -> None:
    if not artifacts:
        lines.append("- explicitly empty for the artifact kinds queried")
        return
    for artifact in artifacts:
        lines.extend(
            (
                f"- artifact_id: {artifact.id}",
                f"  kind: {artifact.kind.value}",
                f"  created_by: {artifact.created_by}",
                f"  created_at: {_timestamp(artifact.created_at)}",
                "  content:",
                f"  {artifact.content}",
            ),
        )


def _append_tasks(
    lines: list[str],
    tasks: tuple[DevelopmentTask, ...],
    policy: AgentContextPolicy,
) -> None:
    if not tasks:
        if policy.include_all_tasks or policy.task_roles:
            lines.append("- explicitly empty for the task scope queried")
        else:
            lines.append("- not requested for this role context policy")
        return
    for task in tasks:
        lines.extend(
            (
                f"- task_id: {task.id}",
                f"  title: {task.title}",
                f"  description: {task.description}",
                f"  assigned_role: {task.assigned_role.value}",
                f"  status: {task.status.value}",
                f"  created_at: {_timestamp(task.created_at)}",
                f"  updated_at: {_timestamp(task.updated_at)}",
            ),
        )


def _timestamp(value: datetime) -> str:
    return value.isoformat()
