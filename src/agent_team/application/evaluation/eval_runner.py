"""Application service for running local evaluation suites."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

from agent_team.application.audit.audit_sanitizer import sanitize_error
from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.application.evaluation.eval_progress_tracker import (
    EvalProgressTracker,
)
from agent_team.application.evaluation.rubric_judge_service import (
    RubricJudgeService,
)
from agent_team.domain.evaluation.candidate_agent_runner import (
    CandidateAgentRunner,
)
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.deterministic_grade import (
    DeterministicGrade,
)
from agent_team.domain.evaluation.eval_attempt_result import (
    EvalAttemptResult,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_clock import EvalClock
from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage
from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_reporter import (
    EvalProgressReporter,
)
from agent_team.domain.evaluation.eval_run_config import EvalRunConfig
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.rubric import Rubric

PACKAGE_VERSION = "0.1.0"
INFRASTRUCTURE_RETRY_BACKOFF_SECONDS = 0.1
RETRYABLE_INFRASTRUCTURE_ERRORS = frozenset(
    {
        "OllamaUnavailableError",
        "WorkflowMCPUnavailableError",
    },
)
MUTATING_TOOL_NAMES = frozenset(
    {
        "add_artifact",
        "apply_patch",
        "create_feature",
        "create_task",
        "update_task_status",
    },
)


@dataclass(frozen=True, slots=True)
class EvalRunner:
    """Run candidate agents against golden datasets."""

    candidate_runner: CandidateAgentRunner
    grader: DeterministicEvalGrader
    judge_service: RubricJudgeService | None = None
    progress_reporter: EvalProgressReporter | None = None
    clock: EvalClock | None = None
    infrastructure_readiness_check: Callable[[], None] | None = None
    infrastructure_retry_backoff_seconds: float = (
        INFRASTRUCTURE_RETRY_BACKOFF_SECONDS
    )

    async def run_suite(
        self,
        suite: EvalSuite,
        rubric: Rubric,
        config: EvalRunConfig,
    ) -> EvalRunResult:
        """Run a suite sequentially and return a local result."""
        started_at = datetime.now(UTC)
        warnings = _warnings(config.candidate_model, config.judge_model)
        case_results: list[EvalCaseResult] = []
        progress = EvalProgressTracker(
            suite=suite,
            config=config,
            reporter=self.progress_reporter,
            clock=self.clock,
        )
        progress.run_started()

        try:
            for case in suite.cases:
                for repetition in range(1, config.repetitions + 1):
                    case_started_at = progress.monotonic()
                    phase_started_at = progress.monotonic()
                    progress.phase_started(
                        case,
                        repetition,
                        EvalPhase.CANDIDATE,
                    )
                    candidate = await _candidate_result_with_retries(
                        eval_runner=self,
                        case=case,
                        repetition=repetition,
                        config=config,
                        progress=progress,
                    )
                    candidate_duration = progress.elapsed_since(
                        phase_started_at,
                    )
                    progress.phase_completed(
                        case,
                        repetition,
                        EvalPhase.CANDIDATE,
                        candidate_duration,
                        include_in_estimate=(
                            not _zero_activity_infrastructure_error(candidate)
                        ),
                    )

                    if _report_as_infrastructure_error(candidate):
                        deterministic_grade = _infrastructure_grade(
                            candidate,
                        )
                        total_case_duration = progress.elapsed_since(
                            case_started_at,
                        )
                        case_results.append(
                            EvalCaseResult(
                                case_id=case.id,
                                repetition=repetition,
                                candidate_result=candidate,
                                deterministic_grade=deterministic_grade,
                                judge_grade=None,
                                verdict=EvalVerdict.INFRASTRUCTURE_ERROR,
                                semantic_judge_required=(
                                    case.semantic_judge_required
                                ),
                                intent=case.intent,
                                context_policy=case.context_policy,
                                candidate_duration_seconds=(
                                    candidate_duration
                                ),
                                deterministic_duration_seconds=None,
                                judge_duration_seconds=None,
                                total_duration_seconds=total_case_duration,
                            ),
                        )
                        progress.case_completed(
                            case,
                            repetition,
                            total_case_duration,
                        )
                        continue

                    phase_started_at = progress.monotonic()
                    progress.phase_started(
                        case,
                        repetition,
                        EvalPhase.DETERMINISTIC_GRADING,
                    )
                    deterministic_grade = self.grader.grade(
                        case,
                        candidate,
                        config.candidate_model,
                    )
                    deterministic_duration = progress.elapsed_since(
                        phase_started_at,
                    )
                    progress.phase_completed(
                        case,
                        repetition,
                        EvalPhase.DETERMINISTIC_GRADING,
                        deterministic_duration,
                    )

                    judge_grade = None
                    judge_duration = None
                    if _should_judge(
                        self.judge_service,
                        config,
                        deterministic_grade.hard_gate_failed,
                        case.semantic_judge_required,
                    ):
                        phase_started_at = progress.monotonic()
                        progress.phase_started(
                            case,
                            repetition,
                            EvalPhase.SEMANTIC_JUDGING,
                        )
                        judge_grade = await _judge_result(
                            service=_require_judge_service(
                                self.judge_service,
                            ),
                            case=case,
                            rubric=rubric,
                            candidate=candidate,
                            config=config,
                        )
                        judge_duration = progress.elapsed_since(
                            phase_started_at,
                        )
                        progress.phase_completed(
                            case,
                            repetition,
                            EvalPhase.SEMANTIC_JUDGING,
                            judge_duration,
                        )

                    verdict = _verdict(
                        deterministic_grade.passed,
                        judge_grade,
                        rubric,
                        case.semantic_judge_required,
                    )
                    total_case_duration = progress.elapsed_since(
                        case_started_at,
                    )
                    case_results.append(
                        EvalCaseResult(
                            case_id=case.id,
                            repetition=repetition,
                            candidate_result=candidate,
                            deterministic_grade=deterministic_grade,
                            judge_grade=judge_grade,
                            verdict=verdict,
                            semantic_judge_required=(
                                case.semantic_judge_required
                            ),
                            intent=case.intent,
                            context_policy=case.context_policy,
                            candidate_duration_seconds=candidate_duration,
                            deterministic_duration_seconds=(
                                deterministic_duration
                            ),
                            judge_duration_seconds=judge_duration,
                            total_duration_seconds=total_case_duration,
                        ),
                    )
                    progress.case_completed(
                        case,
                        repetition,
                        total_case_duration,
                    )
        except asyncio.CancelledError, KeyboardInterrupt:
            progress.run_cancelled()
            raise

        duration_seconds = progress.run_finished()
        ended_at = datetime.now(UTC)
        return EvalRunResult(
            id=str(uuid4()),
            suite_id=suite.id,
            candidate_model=config.candidate_model,
            judge_model=config.judge_model,
            dataset_hash=suite.dataset_hash,
            rubric_hash=rubric.content_hash,
            instructions_hash=config.instructions_hash,
            package_version=PACKAGE_VERSION,
            started_at=started_at,
            ended_at=ended_at,
            case_results=tuple(case_results),
            warnings=warnings,
            case_filter=config.case_id,
            duration_seconds=duration_seconds,
            candidate_thinking_enabled=config.candidate_thinking_enabled,
            judge_thinking_enabled=config.judge_thinking_enabled,
        )


def _warnings(
    candidate_model: str,
    judge_model: str | None,
) -> tuple[str, ...]:
    if judge_model is not None and judge_model == candidate_model:
        return ("candidate and judge model are identical; self-judging bias",)
    if judge_model is None:
        return ("semantic rubric judge was not run",)
    return ()


def _should_judge(
    judge_service: RubricJudgeService | None,
    config: EvalRunConfig,
    hard_gate_failed: bool,
    semantic_judge_required: bool,
) -> bool:
    return (
        judge_service is not None
        and config.judge_model is not None
        and not hard_gate_failed
        and semantic_judge_required
    )


def _require_judge_service(
    judge_service: RubricJudgeService | None,
) -> RubricJudgeService:
    if judge_service is None:
        raise RuntimeError("Judge service is required for judge execution.")
    return judge_service


async def _candidate_result(
    runner: CandidateAgentRunner,
    case: EvalCase,
    candidate_model: str,
    repetition: int,
) -> CandidateRunResult:
    try:
        return await runner.run_case(case, candidate_model, repetition)
    except Exception as error:
        error_type, error_message = sanitize_error(error)
        return CandidateRunResult(
            role=case.active_role,
            model=candidate_model,
            final_response="",
            tool_calls=(),
            database_effects=(),
            status="failed",
            error_type=error_type,
            error_message=error_message,
            error_stage=EvalErrorStage.CANDIDATE_EXECUTION.value,
        )


async def _candidate_result_with_retries(
    eval_runner: EvalRunner,
    case: EvalCase,
    repetition: int,
    config: EvalRunConfig,
    progress: EvalProgressTracker,
) -> CandidateRunResult:
    attempts: list[EvalAttemptResult] = []
    retry_count = 0
    max_attempts = config.infrastructure_retries + 1
    while True:
        attempt_number = len(attempts) + 1
        attempt_started_at = progress.monotonic()
        candidate = await _candidate_result(
            runner=eval_runner.candidate_runner,
            case=case,
            candidate_model=config.candidate_model,
            repetition=repetition,
        )
        attempt_duration = progress.elapsed_since(attempt_started_at)
        attempts.append(
            _attempt_result(
                attempt=attempt_number,
                candidate=candidate,
                duration_seconds=attempt_duration,
            ),
        )

        if not _retryable_infrastructure_error(candidate):
            return _with_attempts(candidate, attempts, retry_count)
        if attempt_number >= max_attempts:
            return _with_attempts(candidate, attempts, retry_count)

        retry_count += 1
        progress.infrastructure_retry(
            case=case,
            repetition=repetition,
            retry_number=retry_count,
            total_retries=config.infrastructure_retries,
        )
        _run_readiness_check(eval_runner.infrastructure_readiness_check)
        await asyncio.sleep(
            _bounded_backoff(eval_runner.infrastructure_retry_backoff_seconds),
        )


def _attempt_result(
    attempt: int,
    candidate: CandidateRunResult,
    duration_seconds: float,
) -> EvalAttemptResult:
    return EvalAttemptResult(
        attempt=attempt,
        status=candidate.status,
        duration_seconds=duration_seconds,
        error_type=candidate.error_type,
        error_stage=candidate.error_stage,
    )


def _with_attempts(
    candidate: CandidateRunResult,
    attempts: list[EvalAttemptResult],
    retry_count: int,
) -> CandidateRunResult:
    return replace(
        candidate,
        attempt_count=len(attempts),
        retry_count=retry_count,
        attempts=tuple(attempts),
    )


def _retryable_infrastructure_error(
    candidate: CandidateRunResult,
) -> bool:
    return (
        _is_infrastructure_error(candidate)
        and candidate.final_response == ""
        and candidate.error_type in RETRYABLE_INFRASTRUCTURE_ERRORS
        and candidate.error_stage
        in {
            EvalErrorStage.CANDIDATE_EXECUTION.value,
            EvalErrorStage.INFRASTRUCTURE_SETUP.value,
        }
        and not candidate.database_effects
        and not _has_reached_mutation(candidate)
    )


def _zero_activity_infrastructure_error(
    candidate: CandidateRunResult,
) -> bool:
    return (
        _is_infrastructure_error(candidate)
        and candidate.final_response == ""
        and not candidate.tool_calls
        and not candidate.skill_calls
        and not candidate.database_effects
        and candidate.error_stage == EvalErrorStage.INFRASTRUCTURE_SETUP.value
    )


def _has_reached_mutation(candidate: CandidateRunResult) -> bool:
    return any(
        tool_call.name in MUTATING_TOOL_NAMES and tool_call.reached_mcp
        for tool_call in candidate.tool_calls
    )


def _run_readiness_check(
    readiness_check: Callable[[], None] | None,
) -> None:
    if readiness_check is not None:
        readiness_check()


def _bounded_backoff(backoff_seconds: float) -> float:
    return min(max(0.0, backoff_seconds), 1.0)


def _is_infrastructure_error(candidate: CandidateRunResult) -> bool:
    return candidate.status == EvalVerdict.INFRASTRUCTURE_ERROR.value


def _report_as_infrastructure_error(candidate: CandidateRunResult) -> bool:
    return (
        _is_infrastructure_error(candidate) and not candidate.database_effects
    )


def _infrastructure_grade(
    candidate: CandidateRunResult,
) -> DeterministicGrade:
    detail = candidate.error_type or "InfrastructureError"
    stage = candidate.error_stage or "unknown"
    return DeterministicGrade(
        passed=False,
        hard_gate_failed=False,
        reasons=(f"infrastructure_error during {stage}: {detail}",),
    )


async def _judge_result(
    service: RubricJudgeService,
    case: EvalCase,
    rubric: Rubric,
    candidate: CandidateRunResult,
    config: EvalRunConfig,
) -> JudgeGrade:
    if config.judge_model is None:
        raise RuntimeError("Judge model is required for judge execution.")
    try:
        return await service.judge_case(
            case,
            rubric,
            candidate,
            config.judge_model,
            config.judge_repetitions,
        )
    except Exception as error:
        error_type, error_message = sanitize_error(error)
        return JudgeGrade(
            verdict=EvalVerdict.JUDGE_ERROR,
            scores={},
            reasons={},
            confidence=0.0,
            ambiguous=True,
            error_message=f"{error_type}: {error_message}",
            judge_model=config.judge_model,
        )


def _verdict(
    deterministic_passed: bool,
    judge_grade: JudgeGrade | None,
    rubric: Rubric,
    semantic_judge_required: bool,
) -> EvalVerdict:
    if not deterministic_passed:
        return EvalVerdict.DETERMINISTIC_FAILED
    if not semantic_judge_required:
        return EvalVerdict.PASSED
    if judge_grade is None:
        return EvalVerdict.NOT_JUDGED
    return _judged_verdict(judge_grade, rubric)


def _judged_verdict(
    judge_grade: JudgeGrade,
    rubric: Rubric,
) -> EvalVerdict:
    if judge_grade.verdict is EvalVerdict.JUDGE_ERROR:
        verdict = EvalVerdict.JUDGE_ERROR
    elif judge_grade.ambiguous:
        verdict = EvalVerdict.AMBIGUOUS
    elif _judge_passed(judge_grade, rubric):
        verdict = EvalVerdict.PASSED
    else:
        verdict = EvalVerdict.JUDGE_FAILED
    return verdict


def _judge_passed(grade: JudgeGrade, rubric: Rubric) -> bool:
    return (
        grade.verdict is EvalVerdict.PASS
        and _critical_scores_pass(grade, rubric)
        and _weighted_score(grade, rubric) >= rubric.threshold
    )


def _critical_scores_pass(grade: JudgeGrade, rubric: Rubric) -> bool:
    return all(
        grade.scores.get(dimension.id, 0) >= dimension.minimum_score
        for dimension in rubric.dimensions
        if dimension.critical
    )


def _weighted_score(grade: JudgeGrade, rubric: Rubric) -> float:
    total_weight = sum(dimension.weight for dimension in rubric.dimensions)
    if total_weight <= 0:
        return 0.0
    weighted_total = sum(
        (grade.scores.get(dimension.id, 0) / 4) * dimension.weight
        for dimension in rubric.dimensions
    )
    return weighted_total / total_weight
