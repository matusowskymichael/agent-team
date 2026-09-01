"""Human-only command-line interface for local evaluations."""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from agent_team.application.audit.audit_sanitizer import sanitize_full_text
from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.application.evaluation.eval_comparison_service import (
    EvalComparisonService,
)
from agent_team.application.evaluation.eval_runner import EvalRunner
from agent_team.application.evaluation.judge_calibration_service import (
    JudgeCalibrationService,
)
from agent_team.application.evaluation.rubric_judge_service import (
    RubricJudgeService,
)
from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.runtime.agent_runtime_instructions import (
    build_runtime_instructions,
)
from agent_team.application.skills.agent_skill_authorizer import (
    AgentSkillAuthorizer,
)
from agent_team.application.skills.agent_skill_context_builder import (
    AgentSkillContextBuilder,
)
from agent_team.application.skills.agent_skill_service import (
    AgentSkillService,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_run_config import EvalRunConfig
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.human_label import HumanLabel
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)
from agent_team.infrastructure.evaluation.eval_hashes import hash_text_value
from agent_team.infrastructure.evaluation.json_eval_result_repository import (
    JsonEvalResultRepository,
)
from agent_team.infrastructure.evaluation.jsonl_golden_dataset_loader import (
    JsonlGoldenDatasetLoader,
)
from agent_team.infrastructure.evaluation.local_candidate_agent_runner import (
    LocalCandidateAgentRunner,
)
from agent_team.infrastructure.evaluation.local_ollama_eval_judge import (
    LocalOllamaEvalJudge,
)
from agent_team.infrastructure.evaluation.markdown_rubric_loader import (
    MarkdownRubricLoader,
)
from agent_team.infrastructure.ollama.ollama_model_catalog import (
    ensure_ollama_model_ready,
)
from agent_team.infrastructure.ollama.ollama_settings import (
    OllamaSettings,
    load_ollama_settings,
)
from agent_team.infrastructure.skills.filesystem_agent_skill_catalog import (
    FilesystemAgentSkillCatalog,
)
from agent_team.interfaces.cli.eval_duration_format import format_duration
from agent_team.interfaces.cli.terminal_eval_progress_reporter import (
    TerminalEvalProgressReporter,
)

EVALS_DIR = Path("evals")
DATASETS_DIR = EVALS_DIR / "datasets"
RUBRICS_DIR = EVALS_DIR / "rubrics"
DEFAULT_RUBRIC = RUBRICS_DIR / "business_analyst.md"
RUBRIC_FILES_BY_ID = {
    "business_analyst_workflow": DEFAULT_RUBRIC,
    "backend_developer_workflow": RUBRICS_DIR / "backend_developer.md",
    "frontend_developer_workflow": RUBRICS_DIR / "frontend_developer.md",
    "software_architect_workflow": RUBRICS_DIR / "software_architect.md",
}
OLLAMA_JUDGE_MODEL_ENV = "OLLAMA_JUDGE_MODEL"
EXIT_SUCCESS = 0
EXIT_QUALITY_FAILURE = 1
EXIT_SYSTEM_ERROR = 2
EXIT_INTERRUPTED = 130


def main(argv: Sequence[str] | None = None) -> int:
    """Run the human-only evaluation CLI."""
    arguments = _parse_arguments(argv)
    try:
        if arguments.command == "list-suites":
            _list_suites()
        elif arguments.command == "run":
            return _exit_code_for_result(asyncio.run(_run(arguments)))
        elif arguments.command == "show":
            _show(
                cast("str", arguments.eval_run_id),
                verbose=cast("bool", arguments.verbose),
            )
        elif arguments.command == "compare":
            _compare(arguments)
        elif arguments.command == "calibrate":
            _calibrate(arguments)
        else:
            raise RuntimeError("Unsupported eval command.")
    except (
        InvalidAgentSkillError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return EXIT_SYSTEM_ERROR
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    return EXIT_SUCCESS


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-team-eval")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("list-suites")

    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("--suite", required=True)
    run_parser.add_argument("--candidate-model")
    run_parser.add_argument("--judge-model")
    run_parser.add_argument("--case-id")
    run_parser.add_argument("--repetitions", type=_positive_integer, default=1)
    run_parser.add_argument(
        "--infrastructure-retries",
        type=_non_negative_integer,
        default=1,
    )
    run_parser.add_argument(
        "--judge-repetitions",
        type=_positive_integer,
        default=1,
    )
    run_parser.add_argument("--no-judge", action="store_true")
    run_parser.add_argument("--no-progress", action="store_true")

    show_parser = subcommands.add_parser("show")
    show_parser.add_argument("eval_run_id")
    show_parser.add_argument("--verbose", action="store_true")

    compare_parser = subcommands.add_parser("compare")
    compare_parser.add_argument("baseline_run_id")
    compare_parser.add_argument("candidate_run_id")
    compare_parser.add_argument(
        "--allow-non-equivalent",
        action="store_true",
    )

    calibrate_parser = subcommands.add_parser("calibrate")
    calibrate_parser.add_argument("--eval-run-id", required=True)
    calibrate_parser.add_argument("--human-labels", required=True)
    return parser.parse_args(argv)


def _list_suites() -> None:
    for path in sorted(DATASETS_DIR.glob("*.jsonl")):
        print(path.stem)


async def _run(arguments: argparse.Namespace) -> EvalRunResult:
    candidate_model = _candidate_model(arguments)
    judge_model = _judge_model(arguments)
    suite = JsonlGoldenDatasetLoader().load(
        suite_id=cast("str", arguments.suite),
        path=DATASETS_DIR / f"{arguments.suite}.jsonl",
    )
    case_id = cast("str | None", arguments.case_id)
    selected_suite = _selected_suite(suite, case_id)

    base_settings = load_ollama_settings(model_override=candidate_model)
    ensure_ollama_model_ready(base_settings)
    if judge_model is not None:
        ensure_ollama_model_ready(
            OllamaSettings(
                base_url=base_settings.base_url,
                model=judge_model,
                max_output_tokens=base_settings.max_output_tokens,
                thinking_enabled=base_settings.thinking_enabled,
            ),
        )

    rubric = MarkdownRubricLoader().load(_rubric_path(selected_suite))
    role = selected_suite.cases[0].active_role
    profile = AgentProfileCatalog().get_profile(role)
    skill_service = _skill_service()
    skill_context = AgentSkillContextBuilder().build_context(
        skill_service.list_available_metadata(profile),
    )
    instructions_hash = hash_text_value(
        build_runtime_instructions(profile, skill_context=skill_context),
    )
    judge_service = None
    if judge_model is not None:
        judge_service = RubricJudgeService(
            LocalOllamaEvalJudge(base_settings),
        )
    progress_reporter = _progress_reporter(arguments)
    runner = EvalRunner(
        candidate_runner=LocalCandidateAgentRunner(base_settings),
        grader=DeterministicEvalGrader(),
        judge_service=judge_service,
        progress_reporter=progress_reporter,
        infrastructure_readiness_check=lambda: ensure_ollama_model_ready(
            base_settings,
        ),
    )
    try:
        result = await runner.run_suite(
            suite=selected_suite,
            rubric=rubric,
            config=EvalRunConfig(
                candidate_model=candidate_model,
                instructions_hash=instructions_hash,
                repetitions=cast("int", arguments.repetitions),
                judge_model=judge_model,
                judge_repetitions=cast("int", arguments.judge_repetitions),
                case_id=case_id,
                infrastructure_retries=cast(
                    "int",
                    arguments.infrastructure_retries,
                ),
                candidate_thinking_enabled=base_settings.thinking_enabled,
                judge_thinking_enabled=None
                if judge_model is None
                else base_settings.thinking_enabled,
            ),
        )
    finally:
        if progress_reporter is not None:
            progress_reporter.close()
    JsonEvalResultRepository().save(result)
    _print_run_summary(result)
    return result


def _selected_suite(suite: EvalSuite, case_id: str | None) -> EvalSuite:
    if case_id is None:
        return suite
    selected_cases = tuple(case for case in suite.cases if case.id == case_id)
    if not selected_cases:
        raise ValueError(f"Case ID {case_id} was not found in {suite.id}.")
    return EvalSuite(
        id=suite.id,
        cases=selected_cases,
        dataset_hash=suite.dataset_hash,
        dataset_version=suite.dataset_version,
    )


def _rubric_path(suite: EvalSuite) -> Path:
    rubric_ids = {case.rubric_id for case in suite.cases}
    if len(rubric_ids) != 1:
        raise ValueError("Evaluation suite must use one rubric.")
    rubric_id = next(iter(rubric_ids))
    rubric_path = RUBRIC_FILES_BY_ID.get(rubric_id)
    if rubric_path is None:
        raise ValueError(f"Unknown evaluation rubric: {rubric_id}.")
    return rubric_path


def _candidate_model(arguments: argparse.Namespace) -> str:
    explicit_model = cast("str | None", arguments.candidate_model)
    return load_ollama_settings(model_override=explicit_model).model


def _judge_model(arguments: argparse.Namespace) -> str | None:
    if cast("bool", arguments.no_judge):
        return None
    explicit_model = cast("str | None", arguments.judge_model)
    configured_model = explicit_model or os.environ.get(OLLAMA_JUDGE_MODEL_ENV)
    if configured_model is None:
        raise ValueError(
            "Judge model is required unless --no-judge is used.",
        )
    return configured_model


def _progress_reporter(
    arguments: argparse.Namespace,
) -> TerminalEvalProgressReporter | None:
    if cast("bool", arguments.no_progress):
        return None
    if not sys.stderr.isatty():
        return None
    return TerminalEvalProgressReporter(sys.stderr)


def _show(eval_run_id: str, verbose: bool = False) -> None:
    result = _require_result(eval_run_id)
    suite = _load_suite_for_result(result)
    cases = _cases_by_id(suite)
    rubric = MarkdownRubricLoader().load(_rubric_path(suite))
    _print_run_summary(result)
    for case_result in result.case_results:
        _print_case_detail(
            case_result,
            cases.get(case_result.case_id),
            rubric,
            verbose,
        )


def _compare(arguments: argparse.Namespace) -> None:
    baseline = _require_result(cast("str", arguments.baseline_run_id))
    candidate = _require_result(cast("str", arguments.candidate_run_id))
    comparison = EvalComparisonService().compare(
        baseline=baseline,
        candidate=candidate,
        allow_mismatched_inputs=cast("bool", arguments.allow_non_equivalent),
    )
    print(f"Baseline: {comparison.baseline_id}")
    print(f"Candidate: {comparison.candidate_id}")
    print(
        "Deterministic improvements: "
        f"{', '.join(comparison.deterministic_improved_cases) or '-'}",
    )
    print(
        "Deterministic regressions: "
        f"{', '.join(comparison.deterministic_regressed_cases) or '-'}",
    )
    print(
        "Deterministic not comparable: "
        f"{', '.join(comparison.deterministic_uncomparable_cases) or '-'}",
    )
    print(
        "Semantic improvements: "
        f"{', '.join(comparison.semantic_improved_cases) or '-'}",
    )
    print(
        "Semantic regressions: "
        f"{', '.join(comparison.semantic_regressed_cases) or '-'}",
    )
    print(
        "Semantic not comparable: "
        f"{', '.join(comparison.semantic_uncomparable_cases) or '-'}",
    )


def _calibrate(arguments: argparse.Namespace) -> None:
    result = _require_result(cast("str", arguments.eval_run_id))
    labels = _load_human_labels(Path(cast("str", arguments.human_labels)))
    calibration = JudgeCalibrationService().calibrate(result, labels)
    print(f"Verdict agreement: {calibration.verdict_agreement:.2f}")
    print(f"Judge ambiguity rate: {calibration.judge_ambiguity_rate:.2f}")
    if calibration.disagreements:
        print(f"Disagreements: {', '.join(calibration.disagreements)}")


def _require_result(eval_run_id: str) -> EvalRunResult:
    result = JsonEvalResultRepository().get(eval_run_id)
    if result is None:
        raise ValueError(f"Evaluation run {eval_run_id} was not found.")
    return result


def _load_human_labels(path: Path) -> tuple[HumanLabel, ...]:
    labels: list[HumanLabel] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError("Human label rows must be JSON objects.")
        data = cast("dict[str, object]", parsed)
        labels.append(
            HumanLabel(
                case_id=_text(data, "case_id"),
                rubric_id=_text(data, "rubric_id"),
                rubric_version=_text(data, "rubric_version"),
                scores=_scores(data.get("scores")),
                verdict=EvalVerdict(_text(data, "verdict")),
                reason=_text(data, "reason"),
                rater=_text(data, "rater"),
                rated_at=_text(data, "rated_at"),
            ),
        )
    return tuple(labels)


def _print_run_summary(result: EvalRunResult) -> None:
    counts = _summary_counts(result)
    print(f"Eval run: {result.id}")
    print(f"Suite: {result.suite_id}")
    print(f"Case filter: {result.case_filter or '-'}")
    print(f"Candidate model: {result.candidate_model}")
    print(f"Candidate thinking: {result.candidate_thinking_enabled}")
    print(f"Judge model: {result.judge_model or '-'}")
    print(f"Judge thinking: {_optional_bool(result.judge_thinking_enabled)}")
    print(f"Duration: {format_duration(result.duration_seconds)}")
    if result.judge_model is None:
        _print_no_judge_summary(counts)
    else:
        print("Results:")
        for key in _CASE_STATUS_KEYS:
            print(f"  {key}: {counts[key]}")
        print(f"  total: {counts['total']}")
    print("Infrastructure:")
    print(f"  retries: {counts['infrastructure_retries']}")
    print(f"  failures: {counts['infrastructure_error']}")
    for warning in result.warnings:
        print(f"Warning: {warning}")


def _print_no_judge_summary(counts: dict[str, int]) -> None:
    deterministic_failed = counts["deterministic_failed"]
    deterministic_passed = (
        counts["total"] - deterministic_failed - counts["infrastructure_error"]
    )
    print("Deterministic:")
    print(f"  passed: {deterministic_passed}")
    print(f"  failed: {deterministic_failed}")
    print("Semantic judge:")
    print(f"  not run: {counts['not_judged']}")
    print(f"  not applicable: {counts['semantic_not_applicable']}")
    print("Overall fully evaluated:")
    print(f"  passed: {counts['passed']}")
    print(
        "  incomplete: "
        f"{counts['not_judged'] + counts['infrastructure_error']}",
    )
    print("Overall:")
    print(f"  infrastructure_error: {counts['infrastructure_error']}")


def _print_case_detail(
    result: EvalCaseResult,
    case: EvalCase | None,
    rubric: Rubric,
    verbose: bool,
) -> None:
    print(f"Case: {result.case_id}")
    print(f"  Repetition: {result.repetition}")
    print(f"  Final status: {_case_status(result)}")
    deterministic = result.deterministic_grade
    deterministic_status = (
        "not run"
        if result.verdict is EvalVerdict.INFRASTRUCTURE_ERROR
        else "passed"
        if deterministic.passed
        else "failed"
    )
    print(
        f"  Deterministic checks: {deterministic_status}",
    )
    print(f"  Hard gate failed: {deterministic.hard_gate_failed}")
    print(
        "  Deterministic failure reasons: "
        f"{'; '.join(deterministic.reasons) or '-'}",
    )
    print(
        "  Expected tool calls: "
        f"{_expected_tool_calls(case) if case is not None else '-'}",
    )
    print(
        f"  Observed tool calls: {_observed_tool_calls(result)}",
    )
    print(
        "  Expected database effects: "
        f"{_expected_database_effects(case) if case is not None else '-'}",
    )
    print(
        f"  Observed database effects: {_observed_database_effects(result)}",
    )
    _print_judge_detail(result)
    if verbose:
        _print_verbose_case_detail(result, case, rubric)


def _print_judge_detail(result: EvalCaseResult) -> None:
    grade = result.judge_grade
    if grade is None:
        status = (
            "not_applicable"
            if not result.semantic_judge_required
            else "not_judged"
        )
        print(f"  Judge status: {status}")
        print("  Judge scores: -")
        return
    print(f"  Judge status: {grade.verdict.value}")
    print(
        f"  Judge scores: {_json(grade.scores) if grade.scores else '-'}",
    )
    if grade.validation_errors:
        errors = "; ".join(grade.validation_errors)
        print(f"  Judge validation errors: {errors}")
    if grade.response_preview:
        print(f"  Judge output preview: {grade.response_preview}")


def _print_verbose_case_detail(
    result: EvalCaseResult,
    case: EvalCase | None,
    rubric: Rubric,
) -> None:
    print("  Candidate final response:")
    print(_indent(sanitize_full_text(result.candidate_result.final_response)))
    print(f"  Case intent: {result.intent.value}")
    print(f"  Evaluation context policy: {result.context_policy.value}")
    print(
        "  Expected tool trajectories: "
        f"{_expected_tool_trajectories(case) if case is not None else '-'}",
    )
    print(
        "  Semantic response requirements: "
        f"{_semantic_requirements(case) if case is not None else '-'}",
    )
    print(
        "  Objective response facts: "
        f"{_objective_facts(case) if case is not None else '-'}",
    )
    print(
        "  Prohibited objective claims: "
        f"{_prohibited_claims(case) if case is not None else '-'}",
    )
    print(f"  Observed tool trajectory: {_observed_tool_trajectory(result)}")
    print(f"  Observed skill calls: {_observed_skill_calls(result)}")
    print(f"  Critical thresholds: {_critical_thresholds(rubric)}")
    print(f"  Final status reason: {_final_status_reason(result)}")
    print(
        "  Durations: "
        f"candidate={format_duration(result.candidate_duration_seconds)}, "
        f"deterministic="
        f"{format_duration(result.deterministic_duration_seconds)}, "
        f"judge={format_duration(result.judge_duration_seconds)}, "
        f"total={format_duration(result.total_duration_seconds)}",
    )
    _print_verbose_judge_detail(result)


def _print_verbose_judge_detail(result: EvalCaseResult) -> None:
    grade = result.judge_grade
    if grade is None:
        return
    print(f"  Judge reasons: {_json(grade.reasons)}")
    print(f"  Judge evidence: {_json(grade.evidence)}")


def _summary_counts(result: EvalRunResult) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(_CASE_STATUS_KEYS, 0)
    counts["semantic_not_applicable"] = 0
    counts["infrastructure_retries"] = 0
    for case_result in result.case_results:
        counts[_case_status(case_result)] += 1
        counts["infrastructure_retries"] += (
            case_result.candidate_result.retry_count
        )
        if not case_result.semantic_judge_required:
            counts["semantic_not_applicable"] += 1
    counts["total"] = len(result.case_results)
    _require_summary_invariant(counts)
    return counts


def _require_summary_invariant(counts: dict[str, int]) -> None:
    total = sum(counts[key] for key in _CASE_STATUS_KEYS)
    if total != counts["total"]:
        raise RuntimeError("Evaluation summary categories do not add up.")


def _case_status(result: EvalCaseResult) -> str:
    verdict = result.verdict
    if verdict is EvalVerdict.INFRASTRUCTURE_ERROR:
        return "infrastructure_error"
    if verdict in {EvalVerdict.PASSED, EvalVerdict.PASS}:
        return "passed"
    if not result.deterministic_grade.passed:
        return "deterministic_failed"
    status_by_verdict = {
        EvalVerdict.NOT_JUDGED: "not_judged",
        EvalVerdict.NOT_RUN: "not_judged",
        EvalVerdict.JUDGE_ERROR: "judge_error",
        EvalVerdict.AMBIGUOUS: "ambiguous",
    }
    return status_by_verdict.get(verdict, "judge_failed")


def _exit_code_for_result(result: EvalRunResult) -> int:
    counts = _summary_counts(result)
    if counts["judge_error"] or counts["infrastructure_error"]:
        return EXIT_SYSTEM_ERROR
    failing = (
        counts["deterministic_failed"]
        + counts["judge_failed"]
        + counts["ambiguous"]
    )
    if failing:
        return EXIT_QUALITY_FAILURE
    return EXIT_SUCCESS


def _load_suite_for_result(result: EvalRunResult) -> EvalSuite:
    return JsonlGoldenDatasetLoader().load(
        suite_id=result.suite_id,
        path=DATASETS_DIR / f"{result.suite_id}.jsonl",
    )


def _cases_by_id(suite: EvalSuite) -> dict[str, EvalCase]:
    return {case.id: case for case in suite.cases}


def _expected_tool_calls(case: EvalCase) -> str:
    values = [
        {
            "name": call.name,
            "arguments_subset": call.arguments_subset,
        }
        for call in case.expected_tool_calls
    ]
    return _json(values)


def _expected_tool_trajectories(case: EvalCase) -> str:
    values = [
        {
            "required_tool_calls": [
                {
                    "name": call.name,
                    "arguments_subset": call.arguments_subset,
                }
                for call in trajectory.required_tool_calls
            ],
            "order_matters": trajectory.order_matters,
            "optional_read_only_tool_calls": list(
                trajectory.optional_read_only_tool_calls,
            ),
            "forbidden_tool_calls": list(trajectory.forbidden_tool_calls),
        }
        for trajectory in case.acceptable_tool_trajectories
    ]
    return _json(values)


def _observed_tool_calls(result: EvalCaseResult) -> str:
    values = [
        {
            "name": call.name,
            "arguments": call.arguments,
            "status": call.status,
            "reached_mcp": call.reached_mcp,
        }
        for call in result.candidate_result.tool_calls
    ]
    return _json(values)


def _observed_tool_trajectory(result: EvalCaseResult) -> str:
    return _json([call.name for call in result.candidate_result.tool_calls])


def _observed_skill_calls(result: EvalCaseResult) -> str:
    values = [
        {
            "tool_name": call.tool_name,
            "skill_name": call.skill_name,
            "status": call.status,
            "content_hash": call.content_hash,
            "resource_name": call.resource_name,
        }
        for call in result.candidate_result.skill_calls
    ]
    return _json(values)


def _expected_database_effects(case: EvalCase) -> str:
    values = [
        {
            "table": effect.table,
            "operation": effect.operation,
            "field_values": effect.field_values,
        }
        for effect in case.expected_database_effects
    ]
    return _json(values)


def _semantic_requirements(case: EvalCase) -> str:
    return _json(list(case.semantic_response_requirements))


def _objective_facts(case: EvalCase) -> str:
    return _json(list(case.objective_response_facts))


def _prohibited_claims(case: EvalCase) -> str:
    return _json(list(case.prohibited_objective_claims))


def _critical_thresholds(rubric: Rubric) -> str:
    values = {
        dimension.id: dimension.minimum_score
        for dimension in rubric.dimensions
        if dimension.critical
    }
    return _json(values)


def _final_status_reason(result: EvalCaseResult) -> str:
    status = _case_status(result)
    if status == "infrastructure_error":
        return (
            result.candidate_result.error_message
            or "infrastructure setup failed before candidate execution"
        )
    if result.deterministic_grade.reasons:
        return "; ".join(result.deterministic_grade.reasons)
    if status == "passed" and not result.semantic_judge_required:
        return "deterministic boundary passed; semantic judge not applicable"
    if result.judge_grade is None:
        return "semantic judge was not run"
    if result.judge_grade.error_message:
        return result.judge_grade.error_message
    return f"final verdict is {status}"


def _indent(value: str) -> str:
    if not value:
        return "    -"
    return "\n".join(f"    {line}" for line in value.splitlines())


def _observed_database_effects(result: EvalCaseResult) -> str:
    values = [
        {
            "table": effect.table,
            "operation": effect.operation,
            "field_values": effect.field_values,
        }
        for effect in result.candidate_result.database_effects
    ]
    return _json(values)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _skill_service() -> AgentSkillService:
    return AgentSkillService(
        catalog=FilesystemAgentSkillCatalog(),
        authorizer=AgentSkillAuthorizer(),
    )


_CASE_STATUS_KEYS = (
    "passed",
    "deterministic_failed",
    "infrastructure_error",
    "not_judged",
    "judge_failed",
    "judge_error",
    "ambiguous",
)


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer",
        ) from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _non_negative_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer",
        ) from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _optional_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _text(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Human label {key} must be a string.")
    return value


def _scores(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("Human label scores must be an object.")
    scores: dict[str, int] = {}
    for key, score in cast("dict[str, object]", value).items():
        if not isinstance(score, int):
            raise ValueError("Human label score must be an integer.")
        scores[key] = score
    return scores


if __name__ == "__main__":
    raise SystemExit(main())
