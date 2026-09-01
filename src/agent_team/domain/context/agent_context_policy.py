"""Agent context policy domain model."""

from dataclasses import dataclass

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind


@dataclass(frozen=True, slots=True)
class AgentContextPolicy:
    """Least-privilege context policy for one role."""

    artifact_kinds: frozenset[ArtifactKind]
    task_roles: frozenset[DevelopmentRole]
    include_all_tasks: bool = False
    max_authoritative_context_chars: int = 20_000
    max_conversation_history_items: int = 20
