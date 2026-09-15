"""MCP tool parameter metadata for workflow tools."""

from typing import Annotated, Literal

from pydantic import Field

from agent_team.application.workflow.workflow_service import (
    INITIAL_TASK_STATUSES,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_handoff_limits import (
    DEFAULT_TASK_HANDOFF_LIMITS,
)
from agent_team.domain.workflow.task_status import TaskStatus

ARTIFACT_KIND_VALUES = ", ".join(kind.value for kind in ArtifactKind)
DEVELOPMENT_ROLE_VALUES = ", ".join(role.value for role in DevelopmentRole)
FEATURE_STATUS_VALUES = ", ".join(status.value for status in FeatureStatus)
TASK_STATUS_VALUES = ", ".join(status.value for status in TaskStatus)
INITIAL_TASK_STATUS_VALUES = ", ".join(
    status.value for status in TaskStatus if status in INITIAL_TASK_STATUSES
)
HANDOFF_LIMITS = DEFAULT_TASK_HANDOFF_LIMITS

FeatureIdParameter = Annotated[
    int,
    Field(
        description="Existing feature ID.",
        ge=1,
    ),
]
TaskIdParameter = Annotated[
    int,
    Field(
        description="Existing development task ID.",
        ge=1,
    ),
]
TitleParameter = Annotated[
    str,
    Field(description="Required non-blank title."),
]
DescriptionParameter = Annotated[
    str,
    Field(description="Required non-blank description."),
]
ArtifactContentParameter = Annotated[
    str,
    Field(description="Required non-blank artifact content."),
]
CreatedByParameter = Annotated[
    str,
    Field(description="Required non-blank name of the creator."),
]
FeatureStatusParameter = Annotated[
    FeatureStatus,
    Field(
        description=f"Feature status. Valid values: {FEATURE_STATUS_VALUES}.",
    ),
]
OptionalFeatureStatusParameter = Annotated[
    FeatureStatus | None,
    Field(
        description=(
            "Optional feature status filter. "
            f"Valid values: {FEATURE_STATUS_VALUES}."
        ),
    ),
]
ArtifactKindParameter = Annotated[
    ArtifactKind,
    Field(
        description=f"Artifact kind. Valid values: {ARTIFACT_KIND_VALUES}.",
    ),
]
DevelopmentRoleParameter = Annotated[
    DevelopmentRole,
    Field(
        description=(
            "Role assigned to the task. "
            f"Valid values: {DEVELOPMENT_ROLE_VALUES}."
        ),
    ),
]
TaskStatusParameter = Annotated[
    TaskStatus,
    Field(
        description=f"Task status. Valid values: {TASK_STATUS_VALUES}.",
    ),
]
InitialTaskStatusParameter = Annotated[
    Literal["pending", "blocked"],
    Field(
        description=(
            "Initial task status. Valid values: "
            f"{INITIAL_TASK_STATUS_VALUES}. Tasks enter verification_pending "
            "only through submit_task_for_verification."
        ),
    ),
]
AgentRunIdParameter = Annotated[
    int,
    Field(
        description="Trusted agent run ID injected by runtime context.",
        ge=1,
    ),
]
AttributionParameter = Annotated[
    str,
    Field(
        description=("Trusted actor attribution injected by runtime context."),
        min_length=1,
        max_length=HANDOFF_LIMITS.item_chars,
    ),
]
WorkspaceIdentityHashParameter = Annotated[
    str,
    Field(
        description=(
            "Trusted workspace identity hash injected by runtime context."
        ),
        min_length=1,
        max_length=HANDOFF_LIMITS.item_chars,
    ),
]
ImplementationSummaryParameter = Annotated[
    str,
    Field(
        description="Concise implementation summary for the handoff.",
        min_length=1,
        max_length=HANDOFF_LIMITS.implementation_summary_chars,
    ),
]
HandoffItemParameter = Annotated[
    str,
    Field(
        min_length=1,
        max_length=HANDOFF_LIMITS.item_chars,
    ),
]
ChangedPathsParameter = Annotated[
    list[HandoffItemParameter],
    Field(
        description=(
            "Workspace-relative changed paths derived from successful patch "
            "audit evidence by trusted runtime context."
        ),
        min_length=1,
        max_length=HANDOFF_LIMITS.changed_path_count,
    ),
]
ReusedSymbolsParameter = Annotated[
    list[HandoffItemParameter],
    Field(
        description="Bounded list of reused symbols referenced in the work.",
        max_length=HANDOFF_LIMITS.reused_symbol_count,
    ),
]
NewSymbolsParameter = Annotated[
    list[HandoffItemParameter],
    Field(
        description="Bounded list of new symbols introduced by the work.",
        max_length=HANDOFF_LIMITS.new_symbol_count,
    ),
]
ChecksAttemptedParameter = Annotated[
    list[HandoffItemParameter],
    Field(
        description="Bounded list of checks attempted before submission.",
        min_length=1,
        max_length=HANDOFF_LIMITS.checks_attempted_count,
    ),
]
ReuseNotesParameter = Annotated[
    str,
    Field(
        description="Concise notes about reused code or patterns.",
        min_length=1,
        max_length=HANDOFF_LIMITS.reuse_notes_chars,
    ),
]
LimitationsParameter = Annotated[
    str,
    Field(
        description="Concise implementation limitations, or 'none'.",
        max_length=HANDOFF_LIMITS.limitations_chars,
    ),
]
NextActionParameter = Annotated[
    str,
    Field(
        description="Concise next action for deterministic verification.",
        min_length=1,
        max_length=HANDOFF_LIMITS.next_action_chars,
    ),
]
