"""Artifact domain model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.workflow.artifact_kind import ArtifactKind


@dataclass(frozen=True, slots=True)
class Artifact:
    """A persisted artifact attached to a development feature."""

    id: int
    feature_id: int
    kind: ArtifactKind
    content: str
    created_by: str
    created_at: datetime
