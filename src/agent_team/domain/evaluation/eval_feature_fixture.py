"""Evaluation feature fixture domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.eval_artifact_fixture import (
    EvalArtifactFixture,
)
from agent_team.domain.evaluation.eval_task_fixture import EvalTaskFixture
from agent_team.domain.workflow.feature_status import FeatureStatus


@dataclass(frozen=True, slots=True)
class EvalFeatureFixture:
    """Feature fixture used to seed one evaluation case."""

    id: int
    title: str
    description: str
    status: FeatureStatus
    artifacts: tuple[EvalArtifactFixture, ...]
    tasks: tuple[EvalTaskFixture, ...]
