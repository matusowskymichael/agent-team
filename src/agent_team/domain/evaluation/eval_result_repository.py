"""Evaluation result repository port."""

from typing import Protocol

from agent_team.domain.evaluation.eval_run_result import EvalRunResult


class EvalResultRepository(Protocol):
    """Persistence port for local evaluation run results."""

    def save(self, result: EvalRunResult) -> None:
        """Persist an evaluation run result."""
        ...

    def get(self, result_id: str) -> EvalRunResult | None:
        """Return a saved evaluation result, if it exists."""
        ...

    def list_ids(self) -> list[str]:
        """Return saved evaluation run IDs."""
        ...
