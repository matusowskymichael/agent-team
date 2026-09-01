"""Evaluation artifact fixture domain model."""

from dataclasses import dataclass

from agent_team.domain.workflow.artifact_kind import ArtifactKind


@dataclass(frozen=True, slots=True)
class EvalArtifactFixture:
    """Artifact fixture used to seed an isolated evaluation database."""

    feature_id: int
    kind: ArtifactKind
    content: str
    created_by: str
