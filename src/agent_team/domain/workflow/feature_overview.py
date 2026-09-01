"""Feature overview domain model."""

from dataclasses import dataclass

from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature


@dataclass(frozen=True, slots=True)
class FeatureOverview:
    """A feature with its attached artifacts and development tasks."""

    feature: Feature
    artifacts: tuple[Artifact, ...]
    tasks: tuple[DevelopmentTask, ...]
