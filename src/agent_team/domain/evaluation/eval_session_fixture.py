"""Evaluation session fixture domain model."""

from dataclasses import dataclass

from agent_team.domain.runtime.development_role import DevelopmentRole


@dataclass(frozen=True, slots=True)
class EvalSessionFixture:
    """Pre-existing feature-scoped session binding for an eval case."""

    session_id: str
    feature_id: int
    role: DevelopmentRole
