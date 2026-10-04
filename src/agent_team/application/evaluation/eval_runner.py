"""Application service for running local evaluation suites."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

from agent_team.application.audit.audit_sanitizer import (
    sanitize_diagnostic_text,
    sanitize_error,
)
from agent_team.application.evaluation.async_eval_deadline import (
    AsyncEvalDeadline,
)
from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.application.evaluation.eval_candidate_state import (
    EvalCandidateState,
)
from agent_team.application.evaluation.eval_progress_tracker import (
    EvalProgressTracker,
)
from agent_team.application.evaluation.eval_run_checkpoint import (
    EvalRunCheckpoint,
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
from agent_team.domain.evaluation.eval_candidate_execution_context import (
    EvalCandidateExecutionContext,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_case_timeout_error import (
    EvalCaseTimeoutError,
)
from agent_team.domain.evaluation.eval_clock import EvalClock
from agent_team.domain.evaluation.eval_deadline import EvalDeadline
from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage
from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_reporter import (
    EvalProgressReporter,
)
from agent_team.domain.evaluation.eval_result_repository import (
    EvalResultRepository,
)
from agent_team.domain.evaluation.eval_run_config import EvalRunConfig
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_run_status import EvalRunStatus
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.runtime.agent_cleanup_timeout_error import (
    AgentCleanupTimeoutError,
)
from agent_team.domain.runtime.agent_provider_timeout_error import (
    AgentProviderTimeoutError,
)
from agent_team.domain.runtime.agent_segment_timeout_error import (
    AgentSegmentTimeoutError,
)
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError

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
        "submit_task_for_verification",
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
    result_repository: EvalResultRepository | None = None
    deadline: EvalDeadline | None = None
    heartbeat_interval_seconds: float = 30.0

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
        checkpoint = EvalRunCheckpoint(
            result=EvalRunResult(
                id=str(uuid4()),
                suite_id=suite.id,
                candidate_model=config.candidate_model,
                judge_model=config.judge_model,
                dataset_hash=suite.dataset_hash,
                rubric_hash=rubric.content_hash,
                instructions_hash=config.instructions_hash,
                package_version=PACKAGE_VERSION,
                started_at=started_at,
                ended_at=started_at,
                case_results=(),
                warnings=warnings,
                case_filter=config.case_id,
                duration_seconds=0.0,
                candidate_thinking_enabled=config.candidate_thinking_enabled,
                judge_thinking_enabled=config.judge_thinking_enabled,
                status=EvalRunStatus.RUNNING,
                case_timeout_seconds=config.case_timeout_seconds,
                runtime_watchdogs=config.runtime_watchdogs,
            ),
            repository=self.result_repository,
        )
        checkpoint.persist()
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
                        progress=progress,
                        checkpoint=checkpoint,
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

                    if candidate.status == "timed_out":
                        case_results.append(
                            _record_timed_out_case(
                                checkpoint,
                                progress,
                                case,
                                repetition,
                                candidate_duration,
                            )
                        )
                        continue

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
                        checkpoint.case_completed(
                            case_results[-1],
                            progress.elapsed_since(progress.run_started_at),
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
                    checkpoint.case_completed(
                        case_results[-1],
                        progress.elapsed_since(progress.run_started_at),
                    )
        except (asyncio.CancelledError, KeyboardInterrupt, Exception) as error:
            _checkpoint_terminal_error(checkpoint, progress, error)
            raise

        return _finish_run_checkpoint(checkpoint, progress, case_results)


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
    context: EvalCandidateExecutionContext,
) -> CandidateRunResult:
    try:
        return await runner.run_case(
            case,
            candidate_model,
            repetition,
            context=context,
        )
    except Exception as error:
        error_type, error_message = sanitize_error(error)
        timed_out = isinstance(
            error,
            AgentProviderTimeoutError
            | AgentSegmentTimeoutError
            | AgentCleanupTimeoutError,
        )
        return CandidateRunResult(
            role=case.active_role,
            model=candidate_model,
            final_response="",
            tool_calls=(),
            database_effects=(),
            status="timed_out" if timed_out else "failed",
            error_type=error_type,
            error_message=sanitize_diagnostic_text(error_message),
            error_stage=EvalErrorStage.CANDIDATE_EXECUTION.value,
            termination_reason=(
                "provider_timeout"
                if isinstance(error, AgentProviderTimeoutError)
                else "segment_timeout"
                if isinstance(error, AgentSegmentTimeoutError)
                else "cleanup_timeout"
                if isinstance(error, AgentCleanupTimeoutError)
                else "stalled"
                if isinstance(error, AgentStalledError)
                else "runtime_error"
            ),
        )


async def _candidate_result_with_retries(
    eval_runner: EvalRunner,
    case: EvalCase,
    repetition: int,
    progress: EvalProgressTracker,
    checkpoint: EvalRunCheckpoint,
) -> CandidateRunResult:
    config = progress.config
    attempts: list[EvalAttemptResult] = []
    retry_count = 0
    max_attempts = config.infrastructure_retries + 1
    while True:
        attempt_number = len(attempts) + 1
        attempt_started_at = progress.monotonic()
        checkpoint.start_attempt(case.id, repetition, attempt_number)
        state = EvalCandidateState(
            checkpoint_callback=checkpoint.candidate_checkpoint,
            clock=progress.monotonic,
        )
        context = EvalCandidateExecutionContext(
            case_timeout_seconds=config.case_timeout_seconds,
            watchdogs=config.runtime_watchdogs,
            liveness_observer=state,
            checkpoint=state.checkpoint,
        )
        heartbeat = (
            None
            if eval_runner.progress_reporter is None
            else asyncio.create_task(
                _candidate_heartbeat(
                    progress,
                    case,
                    repetition,
                    state,
                    attempt_started_at,
                    eval_runner.heartbeat_interval_seconds,
                )
            )
        )
        try:
            deadline = eval_runner.deadline or AsyncEvalDeadline()
            candidate = await deadline.run(
                _candidate_result(
                    runner=eval_runner.candidate_runner,
                    case=case,
                    candidate_model=config.candidate_model,
                    repetition=repetition,
                    context=context,
                ),
                config.case_timeout_seconds,
                config.runtime_watchdogs.cleanup_grace_seconds,
            )
            candidate = _reconcile_candidate_result(candidate, state)
        except EvalCaseTimeoutError as error:
            candidate = _terminated_candidate(
                case,
                config,
                state,
                "timed_out",
                type(error).__name__,
                sanitize_diagnostic_text(error),
                "case_timeout",
            )
        except (asyncio.CancelledError, KeyboardInterrupt) as error:
            candidate = _terminated_candidate(
                case,
                config,
                state,
                "interrupted",
                type(error).__name__,
                "Evaluation interrupted by the user.",
                "cancelled",
            )
            attempts.append(
                _attempt_result(
                    attempt_number,
                    candidate,
                    progress.elapsed_since(attempt_started_at),
                )
            )
            checkpoint.candidate_checkpoint(
                _with_attempts(candidate, attempts, retry_count),
            )
            raise
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
        attempt_duration = progress.elapsed_since(attempt_started_at)
        attempts.append(
            _attempt_result(
                attempt=attempt_number,
                candidate=candidate,
                duration_seconds=attempt_duration,
            ),
        )
        checkpoint.candidate_checkpoint(
            _with_attempts(candidate, attempts, retry_count),
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


async def _candidate_heartbeat(  # noqa: PLR0913, PLR0917
    progress: EvalProgressTracker,
    case: EvalCase,
    repetition: int,
    state: EvalCandidateState,
    started_at: float,
    interval_seconds: float,
) -> None:
    """Keep long active attempts visible and stop under task cancellation."""
    while True:
        await asyncio.sleep(min(30.0, max(0.001, interval_seconds)))
        progress.heartbeat(
            case,
            repetition,
            state.current_snapshot(),
            started_at,
        )


def _terminated_candidate(  # noqa: PLR0913, PLR0917
    case: EvalCase,
    config: EvalRunConfig,
    state: EvalCandidateState,
    status: str,
    error_type: str,
    message: str,
    reason: str,
) -> CandidateRunResult:
    """Preserve safely captured effects while marking uncertain termination."""
    base = state.partial or CandidateRunResult(
        role=case.active_role,
        model=config.candidate_model,
        final_response="",
        tool_calls=(),
        database_effects=(),
    )
    return replace(
        base,
        status=status,
        error_type=error_type,
        error_message=message,
        error_stage=EvalErrorStage.CANDIDATE_EXECUTION.value,
        liveness_snapshot=state.current_snapshot(),
        termination_reason=reason,
    )


def _run_status(case_results: list[EvalCaseResult]) -> EvalRunStatus:
    if any(
        item.candidate_result.status == "timed_out" for item in case_results
    ):
        return EvalRunStatus.TIMED_OUT
    if any(item.candidate_result.status == "failed" for item in case_results):
        return EvalRunStatus.FAILED
    return EvalRunStatus.COMPLETED


def _reconcile_candidate_result(
    candidate: CandidateRunResult, state: EvalCandidateState
) -> CandidateRunResult:
    """Retain captured effects if the boundary subsequently raises."""
    partial = state.partial
    if partial is not None:
        candidate = replace(
            candidate,
            tool_calls=candidate.tool_calls or partial.tool_calls,
            database_effects=(
                candidate.database_effects or partial.database_effects
            ),
            skill_calls=candidate.skill_calls or partial.skill_calls,
            liveness_snapshot=(
                candidate.liveness_snapshot or state.current_snapshot()
            ),
        )
    state.checkpoint(candidate)
    return state.partial or candidate


def _finish_run_checkpoint(
    checkpoint: EvalRunCheckpoint,
    progress: EvalProgressTracker,
    case_results: list[EvalCaseResult],
) -> EvalRunResult:
    """Persist the terminal classification and its individual diagnostics."""
    status = _run_status(case_results)
    terminal = next(
        (
            item.candidate_result
            for item in case_results
            if item.candidate_result.status == status.value
        ),
        None,
    )
    return checkpoint.finish(
        status,
        progress.run_finished(),
        None if terminal is None else terminal.error_type,
        None if terminal is None else terminal.error_message,
    )


def _record_timed_out_case(
    checkpoint: EvalRunCheckpoint,
    progress: EvalProgressTracker,
    case: EvalCase,
    repetition: int,
    duration: float,
) -> EvalCaseResult:
    candidate = checkpoint.result.active_candidate
    if candidate is None:
        raise RuntimeError("A timeout requires a candidate checkpoint.")
    result = EvalCaseResult(
        case_id=case.id,
        repetition=repetition,
        candidate_result=candidate,
        deterministic_grade=DeterministicGrade(
            passed=False,
            hard_gate_failed=False,
            reasons=("Candidate safety deadline reached.",),
        ),
        judge_grade=None,
        verdict=EvalVerdict.TIMED_OUT,
        semantic_judge_required=case.semantic_judge_required,
        intent=case.intent,
        context_policy=case.context_policy,
        candidate_duration_seconds=duration,
        total_duration_seconds=duration,
    )
    checkpoint.case_completed(
        result,
        progress.elapsed_since(progress.run_started_at),
    )
    progress.case_completed(case, repetition, duration)
    return result


def _checkpoint_terminal_error(
    checkpoint: EvalRunCheckpoint,
    progress: EvalProgressTracker,
    error: BaseException,
) -> None:
    if isinstance(error, asyncio.CancelledError | KeyboardInterrupt):
        progress.run_cancelled()
        status = EvalRunStatus.INTERRUPTED
        message = "Evaluation interrupted by the user."
    else:
        status = EvalRunStatus.FAILED
        message = (
            "Evaluation execution failed; inspect saved case diagnostics."
        )
    checkpoint.finish(
        status,
        progress.elapsed_since(progress.run_started_at),
        type(error).__name__,
        message,
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
