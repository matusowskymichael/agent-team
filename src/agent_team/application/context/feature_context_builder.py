"""Application service for authoritative feature context."""

from dataclasses import dataclass, field
from datetime import datetime

from agent_team.application.context.feature_context_render_input import (
    FeatureContextRenderInput,
)
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
from agent_team.domain.workflow.development_task_not_found_error import (
    DevelopmentTaskNotFoundError,
)
from agent_team.domain.workflow.feature_not_found_error import (
    FeatureNotFoundError,
)
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_handoff_limits import (
    DEFAULT_TASK_HANDOFF_LIMITS,
)
from agent_team.domain.workflow.task_verification_evidence import (
    TaskVerificationEvidence,
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
        task_id: int | None = None,
        workspace_identity_hash: str | None = None,
    ) -> AgentContextEnvelope:
        """Build fresh least-privilege context for one feature-scoped run."""
        feature = self.repository.get_feature(feature_id)
        if feature is None:
            raise FeatureNotFoundError(f"Feature {feature_id} was not found.")

        policy = self.policies[role]
        artifacts = self._select_artifacts(feature_id, policy)
        tasks = self._select_tasks(feature_id, policy, task_id)
        latest_handoff = (
            None
            if task_id is None
            else self.repository.latest_task_handoff(task_id)
        )
        latest_verification = (
            None
            if task_id is None
            else self.repository.latest_task_verification(task_id)
        )
        authoritative_context = _render_context(
            FeatureContextRenderInput(
                feature=feature,
                artifacts=artifacts,
                tasks=tasks,
                policy=policy,
                role=role,
                task_id=task_id,
                workspace_identity_hash=workspace_identity_hash,
                latest_handoff=latest_handoff,
                latest_verification=latest_verification,
            ),
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
            task_id=task_id,
            workspace_identity_hash=workspace_identity_hash,
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
        task_id: int | None,
    ) -> tuple[DevelopmentTask, ...]:
        if task_id is not None:
            task = self.repository.get_task(task_id)
            if task is None or task.feature_id != feature_id:
                raise DevelopmentTaskNotFoundError(
                    f"Development task {task_id} was not found.",
                )
            if (
                not policy.include_all_tasks
                and task.assigned_role not in policy.task_roles
            ):
                raise DevelopmentTaskNotFoundError(
                    f"Development task {task_id} was not found.",
                )
            return (task,)
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


def _render_context(render_input: FeatureContextRenderInput) -> str:
    feature = render_input.feature
    lines = [
        "AUTHORITATIVE WORKFLOW CONTEXT",
        "This context is refreshed from the local workflow database for this "
        "run. It outranks conversation history.",
        f"Active role: {render_input.role.value}",
        f"Feature ID: {feature.id}",
        f"Feature title: {feature.title}",
        f"Feature description: {feature.description}",
        f"Feature status: {feature.status.value}",
        f"Feature created_at: {_timestamp(feature.created_at)}",
        f"Feature updated_at: {_timestamp(feature.updated_at)}",
        f"Bound task ID: {_optional_int(render_input.task_id)}",
        "",
        "Artifacts included by role policy:",
    ]
    _append_artifacts(lines, render_input.artifacts)
    lines.extend(
        (
            "",
            "Development tasks included by role policy:",
        ),
    )
    _append_tasks(lines, render_input.tasks, render_input.policy)
    lines.extend(("", "Latest task handoff:"))
    _append_handoff(lines, render_input.latest_handoff)
    lines.extend(("", "Latest verification evidence:"))
    _append_verification(lines, render_input.latest_verification)
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
                (f"  verification_contract: {_verification_contract(task)}"),
                f"  created_at: {_timestamp(task.created_at)}",
                f"  updated_at: {_timestamp(task.updated_at)}",
            ),
        )


def _append_handoff(
    lines: list[str],
    handoff: TaskHandoff | None,
) -> None:
    if handoff is None:
        lines.append("- not requested or no handoff exists for this task")
        return
    limits = DEFAULT_TASK_HANDOFF_LIMITS
    changed_paths = _bounded_items(
        handoff.changed_paths,
        limits.changed_path_count,
    )
    summary = _bounded_text(
        handoff.implementation_summary,
        limits.implementation_summary_chars,
    )
    reused_symbols = _bounded_items(
        handoff.reused_symbols,
        limits.reused_symbol_count,
    )
    new_symbols = _bounded_items(
        handoff.new_symbols,
        limits.new_symbol_count,
    )
    reuse_notes = _bounded_text(
        handoff.reuse_notes,
        limits.reuse_notes_chars,
    )
    checks_attempted = _bounded_items(
        handoff.checks_attempted,
        limits.checks_attempted_count,
    )
    limitations = (
        _bounded_text(handoff.limitations, limits.limitations_chars) or "none"
    )
    next_action = _bounded_text(
        handoff.next_action,
        limits.next_action_chars,
    )
    lines.extend(
        (
            f"- handoff_id: {handoff.id}",
            f"  agent_run_id: {handoff.agent_run_id}",
            f"  submitted_by: {handoff.submitted_by.value}",
            f"  changed_paths: {changed_paths}",
            "  implementation_summary:",
            f"  {summary}",
            f"  reused_symbols: {reused_symbols}",
            f"  new_symbols: {new_symbols}",
            f"  reuse_notes: {reuse_notes}",
            f"  checks_attempted: {checks_attempted}",
            f"  limitations: {limitations}",
            f"  next_action: {next_action}",
            f"  created_at: {_timestamp(handoff.created_at)}",
        ),
    )


def _append_verification(
    lines: list[str],
    evidence: TaskVerificationEvidence | None,
) -> None:
    if evidence is None:
        lines.append("- not requested or no verification evidence exists")
        return
    lines.extend(
        (
            f"- verification_id: {evidence.id}",
            f"  submission_id: {evidence.submission_id}",
            f"  verifier_name: {evidence.verifier_name}",
            f"  outcome: {evidence.outcome.value}",
            (
                "  failure_classification: "
                f"{evidence.failure_classification.value}"
            ),
            f"  feedback: {evidence.feedback}",
            f"  started_at: {_timestamp(evidence.started_at)}",
            f"  ended_at: {_timestamp(evidence.ended_at)}",
            "  checks:",
        ),
    )
    if not evidence.checks:
        lines.append("  - explicitly empty")
        return
    for check in evidence.checks:
        lines.extend(
            (
                f"  - name: {check.name}",
                f"    exit_code: {check.exit_code}",
                f"    timed_out: {check.timed_out}",
                f"    stdout_hash: {check.stdout_hash}",
                f"    stderr_hash: {check.stderr_hash}",
            ),
        )


def _timestamp(value: datetime) -> str:
    return value.isoformat()


def _optional_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _verification_contract(task: DevelopmentTask) -> str:
    contract = task.verification_contract
    if contract is None:
        return "none"
    return (
        f"profile={contract.profile_name}; "
        f"required_checks={', '.join(contract.required_checks)}"
    )


def _bounded_items(values: tuple[str, ...], max_count: int) -> str:
    limits = DEFAULT_TASK_HANDOFF_LIMITS
    if not values:
        return "none"
    shown = tuple(
        _bounded_text(value, limits.item_chars) for value in values[:max_count]
    )
    joined = ", ".join(shown)
    omitted = len(values) - max_count
    if omitted > 0:
        return f"{joined} (+{omitted} omitted)"
    return joined


def _bounded_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    suffix = "... [truncated]"
    prefix_length = max(0, max_chars - len(suffix))
    return f"{value[:prefix_length]}{suffix}"
