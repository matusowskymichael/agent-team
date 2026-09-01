"""MCP tool parameter metadata for workflow tools."""

from typing import Annotated

from pydantic import Field

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus

ARTIFACT_KIND_VALUES = ", ".join(kind.value for kind in ArtifactKind)
DEVELOPMENT_ROLE_VALUES = ", ".join(role.value for role in DevelopmentRole)
FEATURE_STATUS_VALUES = ", ".join(status.value for status in FeatureStatus)
TASK_STATUS_VALUES = ", ".join(status.value for status in TaskStatus)

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
