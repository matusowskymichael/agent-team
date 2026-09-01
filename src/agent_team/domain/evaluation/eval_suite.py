"""Evaluation suite domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.eval_case import EvalCase


@dataclass(frozen=True, slots=True)
class EvalSuite:
    """A named suite of golden evaluation cases."""

    id: str
    cases: tuple[EvalCase, ...]
    dataset_hash: str
    dataset_version: str | None = None
