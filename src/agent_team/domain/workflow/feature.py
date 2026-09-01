"""Feature domain model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.workflow.feature_status import FeatureStatus


@dataclass(frozen=True, slots=True)
class Feature:
    """A development feature tracked by the workflow."""

    id: int
    title: str
    description: str
    status: FeatureStatus
    created_at: datetime
    updated_at: datetime
