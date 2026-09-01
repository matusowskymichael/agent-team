"""Expected typed error domain model."""

from dataclasses import dataclass

from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage


@dataclass(frozen=True, slots=True)
class ExpectedError:
    """A typed error expected by a deterministic boundary case."""

    error_type: str
    stage: EvalErrorStage
    message_fragment: str | None = None
