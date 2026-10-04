"""Evaluation run result domain model."""

from dataclasses import dataclass
from datetime import datetime

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_run_status import EvalRunStatus
from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """Persisted result for a complete evaluation suite run."""

    id: str
    suite_id: str
    candidate_model: str
    judge_model: str | None
    dataset_hash: str
    rubric_hash: str
    instructions_hash: str
    package_version: str
    started_at: datetime
    ended_at: datetime
    case_results: tuple[EvalCaseResult, ...]
    warnings: tuple[str, ...]
    case_filter: str | None = None
    duration_seconds: float | None = None
    candidate_thinking_enabled: bool = False
    judge_thinking_enabled: bool | None = None
    status: EvalRunStatus = EvalRunStatus.COMPLETED
    active_case_id: str | None = None
    active_repetition: int | None = None
    active_attempt: int | None = None
    active_candidate: CandidateRunResult | None = None
    termination_type: str | None = None
    termination_message: str | None = None
    case_timeout_seconds: float | None = None
    runtime_watchdogs: AgentWatchdogSettings | None = None
