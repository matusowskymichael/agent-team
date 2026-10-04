"""Incremental immutable checkpoints for one evaluation run identity."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_result_repository import (
    EvalResultRepository,
)
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_run_status import EvalRunStatus


@dataclass(slots=True)
class EvalRunCheckpoint:
    """Persist partial evidence through the existing result port."""

    result: EvalRunResult
    repository: EvalResultRepository | None = None

    def persist(self) -> None:
        """Save the sanitized checkpoint when persistence is configured."""
        if self.repository is not None:
            self.repository.save(self.result)

    def start_attempt(
        self,
        case_id: str,
        repetition: int,
        attempt: int,
    ) -> None:
        """Record which attempt is active before candidate execution starts."""
        self.result = replace(
            self.result,
            active_case_id=case_id,
            active_repetition=repetition,
            active_attempt=attempt,
            active_candidate=None,
        )
        self.persist()

    def candidate_checkpoint(self, candidate: CandidateRunResult) -> None:
        """Save candidate evidence while temporary audit data still exists."""
        self.result = replace(self.result, active_candidate=candidate)
        self.persist()

    def case_completed(
        self,
        case_result: EvalCaseResult,
        elapsed_seconds: float,
    ) -> None:
        """Persist each fully graded case without waiting for the suite."""
        self.result = replace(
            self.result,
            case_results=(*self.result.case_results, case_result),
            duration_seconds=elapsed_seconds,
            active_candidate=case_result.candidate_result,
        )
        self.persist()

    def finish(
        self,
        status: EvalRunStatus,
        duration_seconds: float,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> EvalRunResult:
        """Persist final status and retain useful active diagnostics."""
        self.result = replace(
            self.result,
            status=status,
            duration_seconds=duration_seconds,
            ended_at=datetime.now(UTC),
            termination_type=error_type,
            termination_message=error_message,
        )
        self.persist()
        return self.result
