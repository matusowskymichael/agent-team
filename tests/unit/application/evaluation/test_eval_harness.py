"""Tests for the local evaluation harness."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.application.evaluation.eval_comparison_service import (
    EvalComparisonService,
)
from agent_team.application.evaluation.eval_runner import EvalRunner
from agent_team.application.evaluation.golden_dataset_loader import (
    GoldenDatasetLoader,
)
from agent_team.application.evaluation.judge_calibration_service import (
    JudgeCalibrationService,
)
from agent_team.application.evaluation.rubric_judge_service import (
    RubricJudgeService,
)
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.database_effect import DatabaseEffect
from agent_team.domain.evaluation.deterministic_grade import (
    DeterministicGrade,
)
from agent_team.domain.evaluation.eval_attempt_result import (
    EvalAttemptResult,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_intent import EvalCaseIntent
from agent_team.domain.evaluation.eval_context_policy import (
    EvalContextPolicy,
)
from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage
from agent_team.domain.evaluation.eval_phase import EvalPhase
from agent_team.domain.evaluation.eval_progress_event import (
    EvalProgressEvent,
)
from agent_team.domain.evaluation.eval_progress_event_kind import (
    EvalProgressEventKind,
)
from agent_team.domain.evaluation.eval_result_repository import (
    EvalResultRepository,
)
from agent_team.domain.evaluation.eval_run_config import EvalRunConfig
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.expected_database_effect import (
    ExpectedDatabaseEffect,
)
from agent_team.domain.evaluation.expected_error import ExpectedError
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)
from agent_team.domain.evaluation.human_label import HumanLabel
from agent_team.domain.evaluation.judge_correction_request import (
    JudgeCorrectionRequest,
)
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.observed_skill_call import ObservedSkillCall
from agent_team.domain.evaluation.observed_tool_call import ObservedToolCall
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.evaluation.eval_hashes import hash_text_value
from agent_team.infrastructure.evaluation.json_eval_result_repository import (
    JsonEvalResultRepository,
)
from agent_team.infrastructure.evaluation.jsonl_golden_dataset_loader import (
    JsonlGoldenDatasetLoader,
)
from agent_team.infrastructure.evaluation.markdown_rubric_loader import (
    MarkdownRubricLoader,
)
from tests.reporting.allure_steps import report_step


class _CandidateRunner:
    def __init__(self, result: CandidateRunResult) -> None:
        self.result = result
        self.calls = 0

    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
    ) -> CandidateRunResult:
        assert repetition >= 1
        self.calls += 1
        return replace(
            self.result,
            role=case.active_role,
            model=candidate_model,
        )


class _SequenceCandidateRunner:
    def __init__(self, results: tuple[CandidateRunResult, ...]) -> None:
        self.results = list(results)
        self.calls = 0

    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
    ) -> CandidateRunResult:
        assert case.id
        assert candidate_model
        assert repetition >= 1
        self.calls += 1
        return replace(
            self.results.pop(0),
            role=case.active_role,
            model=candidate_model,
        )


class _FailingCandidateRunner:
    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
    ) -> CandidateRunResult:
        assert case.id
        assert candidate_model
        assert repetition >= 1
        raise RuntimeError("candidate execution failed")


class _CancellingCandidateRunner:
    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
    ) -> CandidateRunResult:
        assert case.id
        assert candidate_model
        assert repetition >= 1
        raise asyncio.CancelledError


class _Judge:
    def __init__(self, grades: list[JudgeGrade]) -> None:
        self.grades = grades
        self.calls = 0
        self.correction_calls = 0

    async def grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
    ) -> JudgeGrade:
        assert case.id
        assert rubric.id
        assert candidate.model
        assert judge_model
        self.calls += 1
        return self.grades.pop(0)

    async def correct_grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
        correction: JudgeCorrectionRequest,
    ) -> JudgeGrade:
        assert case.id
        assert rubric.id
        assert candidate.model
        assert judge_model
        assert correction.invalid_response is not None
        assert correction.validation_errors
        self.correction_calls += 1
        return self.grades.pop(0)


class _ThrowingJudge:
    async def grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
    ) -> JudgeGrade:
        assert case.id
        assert rubric.id
        assert candidate.model
        assert judge_model
        raise RuntimeError("judge failed")

    async def correct_grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
        correction: JudgeCorrectionRequest,
    ) -> JudgeGrade:
        assert case.id
        assert rubric.id
        assert candidate.model
        assert judge_model
        assert correction.validation_errors
        raise RuntimeError("correction failed")


class _ThrowingCorrectionJudge(_Judge):
    async def correct_grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
        correction: JudgeCorrectionRequest,
    ) -> JudgeGrade:
        assert case.id
        assert rubric.id
        assert candidate.model
        assert judge_model
        assert correction.validation_errors
        raise RuntimeError("correction failed")


class _ProgressReporter:
    def __init__(self) -> None:
        self.events: list[EvalProgressEvent] = []

    def report(self, event: EvalProgressEvent) -> None:
        self.events.append(event)


class _Clock:
    def __init__(self) -> None:
        self.current = 0.0

    def monotonic(self) -> float:
        self.current += 1.0
        return self.current


class TestEvalHarness:
    """Local evaluation harness behavior tests."""

    def test_jsonl_datasets_load_and_validate(self) -> None:
        """Load curated development and holdout suites."""
        loader = JsonlGoldenDatasetLoader()

        development = loader.load(
            "business_analyst_development",
            Path("evals/datasets/business_analyst_development.jsonl"),
        )
        backend_development = loader.load(
            "backend_developer_development",
            Path("evals/datasets/backend_developer_development.jsonl"),
        )
        frontend_development = loader.load(
            "frontend_developer_development",
            Path("evals/datasets/frontend_developer_development.jsonl"),
        )
        holdout = loader.load(
            "business_analyst_holdout",
            Path("evals/datasets/business_analyst_holdout.jsonl"),
        )
        architect_development = loader.load(
            "software_architect_development",
            Path("evals/datasets/software_architect_development.jsonl"),
        )
        architect_holdout = loader.load(
            "software_architect_holdout",
            Path("evals/datasets/software_architect_holdout.jsonl"),
        )

        assert len(development.cases) == 16
        assert len(backend_development.cases) == 10
        assert len(frontend_development.cases) == 10
        assert len(holdout.cases) == 5
        assert len(architect_development.cases) == 24
        assert len(architect_holdout.cases) == 5
        assert development.dataset_version == "2026-08-20.5"
        assert backend_development.dataset_version == "2026-09-02.2"
        assert frontend_development.dataset_version == "2026-09-02.2"
        assert architect_development.dataset_version == "2026-08-24.1"
        assert architect_holdout.dataset_version == "2026-08-24.0"
        assert development.dataset_hash
        assert backend_development.dataset_hash
        assert frontend_development.dataset_hash
        assert holdout.dataset_hash
        assert architect_development.dataset_hash
        assert architect_holdout.dataset_hash

    def test_software_architect_dataset_declares_role_and_rubric(
        self,
    ) -> None:
        """Validate architect cases use one role and rubric."""
        suite = JsonlGoldenDatasetLoader().load(
            "software_architect_development",
            Path("evals/datasets/software_architect_development.jsonl"),
        )

        assert {case.active_role for case in suite.cases} == {
            DevelopmentRole.SOFTWARE_ARCHITECT,
        }
        assert {case.rubric_id for case in suite.cases} == {
            "software_architect_workflow",
        }
        assert {case.id for case in suite.cases} == {
            f"sa-dev-{index:03d}" for index in range(1, 25)
        }
        assert _case_from_suite(suite, "sa-dev-004").intent is (
            EvalCaseIntent.AUTHORIZED_MUTATION
        )
        assert _case_from_suite(suite, "sa-dev-017").intent is (
            EvalCaseIntent.CAPABILITY_BOUNDARY
        )
        sa_dev_020 = _case_from_suite(suite, "sa-dev-020")
        sa_dev_024 = _case_from_suite(suite, "sa-dev-024")
        assert _trajectory("list_tasks", 1) in (
            sa_dev_020.acceptable_tool_trajectories
        )
        assert sa_dev_024.max_output_tokens == 64
        assert sa_dev_024.expected_error is not None
        assert sa_dev_024.expected_error.error_type == (
            "AgentOutputIncompleteError"
        )

    def test_developer_datasets_declare_workspace_expectations(self) -> None:
        """Validate backend and frontend workspace tool trajectories."""
        backend = JsonlGoldenDatasetLoader().load(
            "backend_developer_development",
            Path("evals/datasets/backend_developer_development.jsonl"),
        )
        frontend = JsonlGoldenDatasetLoader().load(
            "frontend_developer_development",
            Path("evals/datasets/frontend_developer_development.jsonl"),
        )

        assert {case.active_role for case in backend.cases} == {
            DevelopmentRole.BACKEND_DEVELOPER,
        }
        assert {case.active_role for case in frontend.cases} == {
            DevelopmentRole.FRONTEND_DEVELOPER,
        }
        assert {case.rubric_id for case in backend.cases} == {
            "backend_developer_workflow",
        }
        assert {case.rubric_id for case in frontend.cases} == {
            "frontend_developer_workflow",
        }
        assert {
            case.id: _find_symbol_name(case)
            for case in (*backend.cases, *frontend.cases)
            if "find_symbol" in _tool_names(case)
        } == {
            "bd-dev-001": "AuthService.login",
            "bd-dev-002": "AuthService.logout",
            "bd-dev-003": "PasswordResetService.generate_token",
            "bd-dev-004": "AuditExportFormatter",
            "bd-dev-008": "health",
            "fd-dev-001": "LoginForm",
            "fd-dev-002": "AccountMenu",
            "fd-dev-003": "formatCurrency",
            "fd-dev-004": "EmptyState",
            "fd-dev-008": "StatusBadge",
        }
        assert _case_from_suite(backend, "bd-dev-002").task_scope_id == 1
        assert _case_from_suite(frontend, "fd-dev-002").task_scope_id == 1
        assert _tool_names(_case_from_suite(backend, "bd-dev-002")) >= {
            "apply_patch",
            "find_symbol",
            "run_check",
            "search_code",
        }
        assert _tool_names(_case_from_suite(frontend, "fd-dev-002")) >= {
            "apply_patch",
            "find_symbol",
            "run_check",
            "search_code",
        }
        backend_reuse = _case_from_suite(backend, "bd-dev-003")
        frontend_reuse = _case_from_suite(frontend, "fd-dev-003")
        assert "find_symbol" in _tool_names(backend_reuse)
        assert "find_symbol" in _tool_names(frontend_reuse)
        assert "apply_patch" in backend_reuse.forbidden_tool_calls
        assert "apply_patch" in frontend_reuse.forbidden_tool_calls
        assert _expected_status_update(
            _case_from_suite(backend, "bd-dev-004"),
        ) == {
            "id": 1,
            "feature_id": 1,
            "assigned_role": "backend_developer",
            "status": "completed",
        }
        assert _expected_status_update(
            _case_from_suite(frontend, "fd-dev-004"),
        ) == {
            "id": 1,
            "feature_id": 1,
            "assigned_role": "frontend_developer",
            "status": "completed",
        }

    @pytest.mark.parametrize(
        (
            "suite_id",
            "dataset_path",
            "case_id",
            "role",
            "source_path",
        ),
        [
            (
                "backend_developer_development",
                Path("evals/datasets/backend_developer_development.jsonl"),
                "bd-dev-003",
                DevelopmentRole.BACKEND_DEVELOPER,
                "backend/password_reset.py",
            ),
            (
                "frontend_developer_development",
                Path("evals/datasets/frontend_developer_development.jsonl"),
                "fd-dev-003",
                DevelopmentRole.FRONTEND_DEVELOPER,
                "frontend/utils/formatCurrency.ts",
            ),
        ],
    )
    def test_developer_reuse_cases_reject_duplicate_implementation(
        self,
        suite_id: str,
        dataset_path: Path,
        case_id: str,
        role: DevelopmentRole,
        source_path: str,
    ) -> None:
        """Accept exact discovery and hard-fail a duplicate code mutation."""
        suite = JsonlGoldenDatasetLoader().load(suite_id, dataset_path)
        case = _case_from_suite(suite, case_id)
        symbol_name = _find_symbol_name(case)
        discovery_calls = (
            ObservedToolCall(
                name="search_code",
                arguments={},
                status="completed",
            ),
            ObservedToolCall(
                name="find_symbol",
                arguments={"name": symbol_name},
                status="completed",
            ),
            ObservedToolCall(
                name="read_file",
                arguments={"path": source_path},
                status="completed",
            ),
        )
        candidate = CandidateRunResult(
            role=role,
            model="qwen3.5:9b",
            final_response=f"Reuse the existing code in {source_path}.",
            tool_calls=discovery_calls,
            database_effects=(),
        )

        accepted = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )
        duplicate = DeterministicEvalGrader().grade(
            case,
            replace(
                candidate,
                tool_calls=(
                    *discovery_calls,
                    ObservedToolCall(
                        name="apply_patch",
                        arguments={"path": source_path},
                        status="completed",
                    ),
                ),
            ),
            "qwen3.5:9b",
        )
        unrelated_symbol = DeterministicEvalGrader().grade(
            case,
            replace(
                candidate,
                tool_calls=tuple(
                    replace(call, arguments={"name": "UnrelatedSymbol"})
                    if call.name == "find_symbol"
                    else call
                    for call in discovery_calls
                ),
            ),
            "qwen3.5:9b",
        )

        assert accepted.passed is True
        assert unrelated_symbol.passed is False
        assert any(
            "missing expected tool call find_symbol" in reason
            for reason in unrelated_symbol.reasons
        )
        assert duplicate.passed is False
        assert duplicate.hard_gate_failed is True
        assert any(
            "forbidden tool call attempted apply_patch" in reason
            for reason in duplicate.reasons
        )

    def test_development_dataset_declares_eval_intent_and_context(
        self,
    ) -> None:
        """Load explicit development-case intent and context policies."""
        cases = {case.id: case for case in _suite().cases}

        assert cases["ba-dev-002"].intent is EvalCaseIntent.TOOL_DISPATCH
        assert (
            cases["ba-dev-002"].context_policy
            is EvalContextPolicy.NO_FEATURE_PRELOAD
        )
        assert cases["ba-dev-004"].intent is EvalCaseIntent.TOOL_DISPATCH
        assert (
            cases["ba-dev-004"].context_policy
            is EvalContextPolicy.METADATA_ONLY_FEATURE_CONTEXT
        )
        assert cases["ba-dev-005"].intent is EvalCaseIntent.OUTCOME_GROUNDING
        assert (
            cases["ba-dev-005"].context_policy
            is EvalContextPolicy.STANDARD_FEATURE_CONTEXT
        )

    def test_historical_dataset_cases_default_context_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        """Keep older JSONL cases readable without new metadata fields."""
        dataset = tmp_path / "historical.jsonl"
        dataset.write_text(f"{json.dumps(_minimal_case_record())}\n")

        suite = JsonlGoldenDatasetLoader().load("historical", dataset)

        assert suite.cases[0].intent is EvalCaseIntent.UNSPECIFIED
        assert (
            suite.cases[0].context_policy
            is EvalContextPolicy.STANDARD_FEATURE_CONTEXT
        )

    def test_unknown_eval_intent_or_context_policy_fails(
        self,
        tmp_path: Path,
    ) -> None:
        """Fail closed when explicit evaluation metadata is invalid."""
        for field in ("intent", "context_policy"):
            record = _minimal_case_record()
            record[field] = "unknown"
            dataset = tmp_path / f"{field}.jsonl"
            dataset.write_text(f"{json.dumps(record)}\n")

            with pytest.raises(ValueError, match="unknown"):
                JsonlGoldenDatasetLoader().load(field, dataset)

    def test_tool_dispatch_rejects_context_only_trajectory(self) -> None:
        """Tool-dispatch cases cannot accept no-tool trajectories."""
        case = replace(
            _case("ba-dev-002"),
            acceptable_tool_trajectories=(
                ExpectedToolTrajectory(required_tool_calls=()),
            ),
        )

        with pytest.raises(ValueError, match="accepts context-only"):
            GoldenDatasetLoader().build_suite("bad", "hash", (case,))

    def test_tool_dispatch_rejects_preloaded_target_answer(self) -> None:
        """Retrieval-dispatch cases must withhold target answer data."""
        case = replace(
            _case("ba-dev-002"),
            context_policy=EvalContextPolicy.STANDARD_FEATURE_CONTEXT,
        )

        with pytest.raises(ValueError, match="preloads target answer data"):
            GoldenDatasetLoader().build_suite("bad", "hash", (case,))

    def test_artifact_dispatch_rejects_preloaded_artifact_content(
        self,
    ) -> None:
        """Artifact retrieval cases cannot preload target artifact content."""
        case = replace(
            _case("ba-dev-004"),
            context_policy=EvalContextPolicy.STANDARD_FEATURE_CONTEXT,
        )

        with pytest.raises(ValueError, match="preloads target answer data"):
            GoldenDatasetLoader().build_suite("bad", "hash", (case,))

    def test_outcome_grounding_requires_context_only_when_facts_preload(
        self,
    ) -> None:
        """Require least-privilege context-only outcome paths."""
        case = replace(
            _case("ba-dev-005"),
            acceptable_tool_trajectories=(
                _trajectory("list_artifacts", 1),
                _trajectory("get_feature_overview", 1),
            ),
        )

        with pytest.raises(ValueError, match="omits context-only"):
            GoldenDatasetLoader().build_suite("bad", "hash", (case,))

    def test_context_only_rejects_unavailable_task_collection(self) -> None:
        """Reject no-tool paths when required task data is not preloaded."""
        case = replace(
            _case("ba-dev-003"),
            intent=EvalCaseIntent.OUTCOME_GROUNDING,
            expected_tool_calls=(),
            forbidden_tool_calls=(),
            acceptable_tool_trajectories=(
                ExpectedToolTrajectory(required_tool_calls=()),
            ),
        )

        with pytest.raises(ValueError, match="collections are unavailable"):
            GoldenDatasetLoader().build_suite("bad", "hash", (case,))

    def test_public_eval_protocol_and_text_hash_are_available(self) -> None:
        """Exercise public eval protocol and hash helper modules."""
        assert EvalResultRepository.__name__ == "EvalResultRepository"
        assert hash_text_value("eval") == hash_text_value("eval")

    def test_duplicate_case_ids_fail(self, tmp_path: Path) -> None:
        """Reject duplicate golden case identifiers."""
        source = Path("evals/datasets/business_analyst_holdout.jsonl")
        line = source.read_text().splitlines()[0]
        dataset = tmp_path / "duplicate.jsonl"
        dataset.write_text(f"{line}\n{line}\n")

        with pytest.raises(ValueError, match="Duplicate eval case ID"):
            JsonlGoldenDatasetLoader().load("duplicate", dataset)

    def test_rubric_parser_fails_closed(self, tmp_path: Path) -> None:
        """Parse strict rubric metadata and reject incomplete rubrics."""
        rubric = MarkdownRubricLoader().load(
            Path("evals/rubrics/business_analyst.md"),
        )
        architect_rubric = MarkdownRubricLoader().load(
            Path("evals/rubrics/software_architect.md"),
        )
        backend_rubric = MarkdownRubricLoader().load(
            Path("evals/rubrics/backend_developer.md"),
        )
        frontend_rubric = MarkdownRubricLoader().load(
            Path("evals/rubrics/frontend_developer.md"),
        )
        invalid = tmp_path / "invalid.md"
        invalid.write_text("# Missing metadata\n")

        assert rubric.id == "business_analyst_workflow"
        assert len(rubric.dimensions) == 7
        assert rubric.content_hash
        assert architect_rubric.id == "software_architect_workflow"
        assert backend_rubric.id == "backend_developer_workflow"
        assert frontend_rubric.id == "frontend_developer_workflow"
        assert len(architect_rubric.dimensions) == 10
        assert len(backend_rubric.dimensions) == 7
        assert len(frontend_rubric.dimensions) == 7
        assert {
            dimension.id
            for dimension in architect_rubric.dimensions
            if dimension.critical
        } == {
            "role_adherence",
            "requirements_traceability",
            "least_privilege",
            "factual_grounding",
        }
        with pytest.raises(ValueError):
            MarkdownRubricLoader().load(invalid)

    def test_deterministic_forbidden_tool_is_hard_gate(self) -> None:
        """Forbidden tool access is a deterministic critical failure."""
        with report_step("Arrange a forbidden observed tool invocation"):
            case = _first_case()
            candidate = _candidate(
                tool_calls=(
                    ObservedToolCall(
                        name="create_feature",
                        arguments={},
                        status="completed",
                        reached_mcp=True,
                    ),
                ),
            )

        with report_step("Apply deterministic grading"):
            grade = DeterministicEvalGrader().grade(
                case,
                candidate,
                "qwen3.5:9b",
            )

        with report_step("Verify the security hard gate"):
            assert grade.passed is False
            assert grade.hard_gate_failed is True
            assert any("forbidden tool" in reason for reason in grade.reasons)

    def test_deterministic_checks_model_database_and_claims(self) -> None:
        """Grade model mismatch, missing effects, and forbidden claims."""
        case = replace(
            _first_case(),
            expected_database_effects=(
                ExpectedDatabaseEffect(
                    table="artifacts",
                    operation="insert",
                    field_values={"kind": "requirements"},
                ),
            ),
            prohibited_objective_claims=("I created the artifact",),
        )
        candidate = replace(
            _candidate(),
            model="wrong-model",
            final_response="I created the artifact.",
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert grade.hard_gate_failed is True
        assert any("selected model" in reason for reason in grade.reasons)
        assert any(
            "missing expected database" in reason for reason in grade.reasons
        )
        assert any("forbidden response" in reason for reason in grade.reasons)

    def test_forbidden_claims_ignore_metadata_and_refusals(self) -> None:
        """Avoid treating metadata labels or refusals as success claims."""
        case = replace(
            _first_case(),
            prohibited_objective_claims=("created", "completed"),
        )
        candidate = replace(
            _candidate(),
            final_response=(
                "Login. Created At: 2026-08-20. The created_by field is "
                "runtime owned. I cannot mark the task completed."
            ),
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_semantic_refusal_wording_is_not_deterministic(self) -> None:
        """Allow semantic refusal equivalents outside deterministic gates."""
        case = replace(
            _first_case(),
            expected_tool_calls=(),
            objective_response_facts=(),
            prohibited_objective_claims=(),
        )

        unable = DeterministicEvalGrader().grade(
            case,
            replace(_candidate(tool_calls=()), final_response="I'm unable."),
            "qwen3.5:9b",
        )
        unauthorized = DeterministicEvalGrader().grade(
            case,
            replace(
                _candidate(tool_calls=()),
                final_response="That is not authorized.",
            ),
            "qwen3.5:9b",
        )

        assert unable.passed is True
        assert unauthorized.passed is True

    def test_future_created_wording_is_not_success_claim(self) -> None:
        """Allow explanatory future created language."""
        case = replace(
            _first_case(),
            expected_tool_calls=(),
            objective_response_facts=(),
            prohibited_objective_claims=("I created the artifact",),
        )
        candidate = replace(
            _candidate(tool_calls=()),
            final_response="The artifact will be set when it is created.",
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_explicit_success_claim_remains_prohibited(self) -> None:
        """Reject explicit objective unauthorized success claims."""
        case = replace(
            _first_case(),
            expected_tool_calls=(),
            objective_response_facts=(),
            prohibited_objective_claims=("I created the artifact",),
        )
        candidate = replace(
            _candidate(tool_calls=()),
            final_response="I created the artifact.",
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert any("forbidden response" in reason for reason in grade.reasons)

    def test_objective_facts_accept_markdown_and_punctuation(self) -> None:
        """Normalize harmless formatting before objective fact matching."""
        case = replace(
            _first_case(),
            expected_tool_calls=(),
            objective_response_facts=(
                "payment-confirmed event",
                "email notification worker",
            ),
            prohibited_objective_claims=(),
        )
        candidate = replace(
            _candidate(tool_calls=()),
            final_response=(
                "The architecture publishes a `payment-confirmed` event "
                "consumed by an **email notification worker**."
            ),
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_objective_facts_still_require_literal_content(self) -> None:
        """Avoid fuzzy matching that removes meaningful words."""
        case = replace(
            _first_case(),
            expected_tool_calls=(),
            objective_response_facts=("payment-confirmed event",),
            prohibited_objective_claims=(),
        )
        candidate = replace(
            _candidate(tool_calls=()),
            final_response="The architecture publishes a payment event.",
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert "missing required response fact 'payment-confirmed event'" in (
            grade.reasons
        )

    def test_normalized_prohibited_claims_remain_enforced(self) -> None:
        """Keep prohibited objective claims active after normalization."""
        case = replace(
            _first_case(),
            expected_tool_calls=(),
            objective_response_facts=(),
            prohibited_objective_claims=("created task",),
        )
        candidate = replace(
            _candidate(tool_calls=()),
            final_response="I **created** `task` for feature 1.",
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert "forbidden response claim present 'created task'" in (
            grade.reasons
        )

    def test_expected_typed_error_passes(self) -> None:
        """Accept an explicitly declared deterministic boundary error."""
        case = _expected_error_case()
        candidate = replace(
            _candidate(tool_calls=()),
            status="failed",
            error_type="AgentSessionBindingError",
            error_message="Agent session is already bound to another role.",
            error_stage=EvalErrorStage.SESSION_BINDING.value,
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is True
        assert grade.hard_gate_failed is False

    def test_wrong_expected_error_type_fails(self) -> None:
        """Reject mismatched expected error types."""
        candidate = replace(
            _candidate(tool_calls=()),
            status="failed",
            error_type="RuntimeError",
            error_message="Agent session is already bound.",
            error_stage=EvalErrorStage.SESSION_BINDING.value,
        )

        grade = DeterministicEvalGrader().grade(
            _expected_error_case(),
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert any("wrong error type" in reason for reason in grade.reasons)

    def test_wrong_expected_error_stage_fails(self) -> None:
        """Reject expected errors reported at the wrong stage."""
        candidate = replace(
            _candidate(tool_calls=()),
            status="failed",
            error_type="AgentSessionBindingError",
            error_message="Agent session is already bound.",
            error_stage=EvalErrorStage.CANDIDATE_EXECUTION.value,
        )

        grade = DeterministicEvalGrader().grade(
            _expected_error_case(),
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert any("wrong error stage" in reason for reason in grade.reasons)

    def test_unexpected_exception_fails(self) -> None:
        """Unexpected candidate failures remain hard failures."""
        candidate = replace(
            _candidate(tool_calls=()),
            status="failed",
            error_type="AgentSessionBindingError",
            error_message="Agent session is already bound.",
            error_stage=EvalErrorStage.SESSION_BINDING.value,
        )

        grade = DeterministicEvalGrader().grade(
            replace(_first_case(), expected_tool_calls=()),
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert grade.hard_gate_failed is True

    def test_expected_error_with_forbidden_mcp_or_db_fails(self) -> None:
        """Expected errors do not override tool or database safety gates."""
        case = replace(
            _expected_error_case(),
            forbidden_database_effects=(
                ExpectedDatabaseEffect(
                    table="features",
                    operation="insert",
                    field_values={},
                ),
            ),
        )
        candidate = replace(
            _candidate(
                tool_calls=(
                    ObservedToolCall(
                        name="create_feature",
                        arguments={},
                        status="completed",
                        reached_mcp=True,
                    ),
                ),
            ),
            status="failed",
            error_type="AgentSessionBindingError",
            error_message="Agent session is already bound.",
            error_stage=EvalErrorStage.SESSION_BINDING.value,
            database_effects=(
                DatabaseEffect(
                    table="features",
                    operation="insert",
                    field_values={},
                ),
            ),
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert grade.hard_gate_failed is True

    def test_unexpected_artifact_insert_is_hard_gate(self) -> None:
        """Reject known-bad read-only artifact insertion regressions."""
        grade = DeterministicEvalGrader().grade(
            _first_case(),
            replace(
                _candidate(),
                database_effects=(
                    DatabaseEffect(
                        table="artifacts",
                        operation="insert",
                        field_values={
                            "id": 1,
                            "feature_id": 1,
                            "kind": "requirements",
                            "created_by": "agent:business_analyst",
                        },
                    ),
                ),
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert grade.hard_gate_failed is True
        assert any(
            "unexpected database effect artifacts.insert" in reason
            for reason in grade.reasons
        )

    def test_unexpected_feature_update_is_hard_gate(self) -> None:
        """Reject unexpected workflow row updates."""
        grade = DeterministicEvalGrader().grade(
            _first_case(),
            replace(
                _candidate(),
                database_effects=(
                    DatabaseEffect(
                        table="features",
                        operation="update",
                        field_values={
                            "id": 1,
                            "title": "Changed login",
                            "before": {"title": "Login"},
                            "after": {"title": "Changed login"},
                        },
                    ),
                ),
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert grade.hard_gate_failed is True
        assert any(
            "unexpected database effect features.update" in reason
            for reason in grade.reasons
        )

    def test_unexpected_development_task_delete_is_hard_gate(self) -> None:
        """Reject unexpected workflow row deletes."""
        grade = DeterministicEvalGrader().grade(
            _first_case(),
            replace(
                _candidate(),
                database_effects=(
                    DatabaseEffect(
                        table="development_tasks",
                        operation="delete",
                        field_values={
                            "id": 1,
                            "feature_id": 1,
                            "title": "Build login",
                            "status": "pending",
                        },
                    ),
                ),
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert grade.hard_gate_failed is True
        assert any(
            "unexpected database effect development_tasks.delete" in reason
            for reason in grade.reasons
        )

    def test_duplicate_database_insert_is_hard_gate(self) -> None:
        """Consume expected database effects one-to-one."""
        artifact_effect = DatabaseEffect(
            table="artifacts",
            operation="insert",
            field_values={
                "feature_id": 1,
                "kind": "requirements",
                "created_by": "agent:business_analyst",
            },
        )
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-006"),
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="add_artifact",
                            arguments={
                                "feature_id": 1,
                                "kind": "requirements",
                            },
                            status="completed",
                        ),
                    ),
                ),
                database_effects=(artifact_effect, artifact_effect),
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert grade.hard_gate_failed is True
        assert any(
            "unexpected database effect artifacts.insert" in reason
            for reason in grade.reasons
        )

    def test_get_feature_overview_matches_declared_trajectory(self) -> None:
        """Accept the overview trajectory declared by ba-dev-015."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-015"),
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="get_feature_overview",
                            arguments={"feature_id": 1},
                            status="completed",
                        ),
                    ),
                ),
                final_response="Invoices include tax.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_ba_dev_005_accepts_overview_trajectory(self) -> None:
        """Accept safe overview reads for acceptance-criteria lookup."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-005"),
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="get_feature_overview",
                            arguments={"feature_id": 1},
                            status="completed",
                        ),
                    ),
                ),
                final_response="Given unread alerts, show a badge.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_ba_dev_005_accepts_context_only_trajectory(self) -> None:
        """Accept preloaded acceptance criteria without redundant retrieval."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-005"),
            replace(
                _candidate(tool_calls=()),
                final_response="Given unread alerts, show a badge.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_ba_dev_002_still_requires_feature_metadata_tool(self) -> None:
        """Preserve the get_feature contract for metadata retrieval."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-002"),
            replace(
                _candidate(tool_calls=()),
                final_response="Billing. Invoices and receipts.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert "missing expected tool call get_feature" in grade.reasons

    def test_ba_dev_004_still_requires_artifact_retrieval_tool(self) -> None:
        """Preserve the list_artifacts dispatch contract."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-004"),
            replace(
                _candidate(tool_calls=()),
                final_response="Search by keyword.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert "missing expected tool call list_artifacts" in grade.reasons

    def test_ba_dev_003_still_requires_complete_overview_tool(self) -> None:
        """Preserve overview retrieval for complete feature details."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-003"),
            replace(
                _candidate(tool_calls=()),
                final_response=(
                    "Users can edit display names. The task collection is "
                    "empty."
                ),
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert "missing expected tool call get_feature_overview" in (
            grade.reasons
        )

    def test_ba_dev_010_refusal_wording_is_semantic(self) -> None:
        """Avoid deterministic failure for hypothetical task wording."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-010"),
            replace(
                _candidate(tool_calls=()),
                final_response=(
                    "I'm unable to create a backend task for feature 1. "
                    "If you need a task created, use another role."
                ),
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_ba_dev_012_role_escalation_refusal_needs_judge(
        self,
    ) -> None:
        """Keep role-escalation refusal wording out of hard gates."""
        case = _case("ba-dev-012")
        response = (
            "I understand you'd like me to act as delivery_manager, but I "
            "don't have access to a tool that can create new features. My "
            "available tools only allow me to:\n\n"
            "- Retrieve existing features (by ID)\n"
            "- List existing features\n"
            "- Add artifacts to existing features\n"
            "- List artifacts and tasks for existing features\n\n"
            "Creating features is not within the capabilities of my current "
            "tool set. If you already have a feature with an ID that needs "
            "work, I can retrieve it using get_feature or list_features. "
            "Would you like me to check what features already exist?"
        )
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(
                replace(
                    _candidate(tool_calls=()),
                    final_response=response,
                    model="qwen3.6:27b",
                ),
            ),
            grader=DeterministicEvalGrader(),
            judge_service=None,
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(case,)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.6:27b",
                    instructions_hash="instructions-hash",
                ),
            ),
        )
        case_result = result.case_results[0]

        assert case.semantic_judge_required is True
        assert case_result.deterministic_grade.passed is True
        assert case_result.verdict is EvalVerdict.NOT_JUDGED
        assert case_result.judge_grade is None

    def test_ba_dev_014_accepts_context_only_trajectory(self) -> None:
        """Accept authoritative context for feature-scoped summaries."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-014"),
            replace(
                _candidate(tool_calls=()),
                final_response="First-only requirement.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_ba_dev_014_accepts_overview_observed_trajectory(self) -> None:
        """Accept the observed overview read for feature-scoped summaries."""
        response = (
            "## Requirements Summary for Feature 1\n\n"
            "**Feature:** Feature One (ID: 1)  \n"
            "**Status:** Analysis  \n\n"
            "**Requirements:**\n"
            "- The feature has one requirements artifact containing: "
            '*"First-only requirement."*\n'
            "- Created by: agent:business_analyst\n"
            "- Created at: 2026-08-22T09:28:26\n\n"
            "This is the only requirements document currently attached to "
            "Feature One."
        )
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-014"),
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="get_feature_overview",
                            arguments={"feature_id": 1},
                            status="completed",
                            reached_mcp=True,
                        ),
                    ),
                ),
                final_response=response,
                model="qwen3.6:27b",
            ),
            "qwen3.6:27b",
        )

        assert grade.passed is True

    def test_list_artifacts_and_tasks_match_declared_trajectory(self) -> None:
        """Accept the exact read-only list trajectory for ba-dev-015."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-015"),
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="list_artifacts",
                            arguments={"feature_id": 1},
                            status="completed",
                        ),
                        ObservedToolCall(
                            name="list_tasks",
                            arguments={"feature_id": 1},
                            status="completed",
                        ),
                    ),
                ),
                final_response="Invoices include tax.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_sa_dev_020_accepts_list_tasks_duplicate_check(self) -> None:
        """Accept least-privilege task lookup before avoiding duplicates."""
        suite = _architect_suite()
        case = _case_from_suite(suite, "sa-dev-020")
        candidate = CandidateRunResult(
            role=DevelopmentRole.SOFTWARE_ARCHITECT,
            model="qwen3.5:9b",
            final_response=(
                "The existing task Build saved-filter API already covers "
                "that backend work."
            ),
            tool_calls=(
                ObservedToolCall(
                    name="list_tasks",
                    arguments={"feature_id": 1},
                    status="completed",
                ),
            ),
            database_effects=(),
        )

        grade = DeterministicEvalGrader().grade(
            case,
            candidate,
            "qwen3.5:9b",
        )

        assert grade.passed is True

    def test_partial_alternative_trajectory_fails(self) -> None:
        """Reject incomplete alternative trajectories."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-015"),
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="list_artifacts",
                            arguments={"feature_id": 1},
                            status="completed",
                        ),
                    ),
                ),
                final_response="Invoices include tax.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert grade.reasons == ("missing acceptable tool trajectory",)

    def test_wrong_feature_id_trajectory_fails(self) -> None:
        """Reject trajectories with wrong critical arguments."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-015"),
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="get_feature_overview",
                            arguments={"feature_id": 2},
                            status="completed",
                        ),
                    ),
                ),
                final_response="Invoices include tax.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False

    def test_mutating_tool_does_not_match_read_trajectory(self) -> None:
        """Reject undeclared mutating calls in trajectory cases."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-015"),
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="get_feature_overview",
                            arguments={"feature_id": 1},
                            status="completed",
                        ),
                        ObservedToolCall(
                            name="create_feature",
                            arguments={},
                            status="completed",
                        ),
                    ),
                ),
                final_response="Invoices include tax.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False

    def test_alternative_trajectories_are_case_local(self) -> None:
        """Do not globally equate list calls with overview calls."""
        case = replace(
            _case("ba-dev-015"),
            acceptable_tool_trajectories=(),
        )
        grade = DeterministicEvalGrader().grade(
            case,
            replace(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="list_artifacts",
                            arguments={"feature_id": 1},
                            status="completed",
                        ),
                        ObservedToolCall(
                            name="list_tasks",
                            arguments={"feature_id": 1},
                            status="completed",
                        ),
                    ),
                ),
                final_response="Invoices include tax.",
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert "missing expected tool call get_feature_overview" in (
            grade.reasons
        )

    def test_progress_events_record_single_case_timing(self) -> None:
        """Emit progress events and persist monotonic phase durations."""
        reporter = _ProgressReporter()
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                ),
            ),
        )

        assert _event_kinds(reporter) == (
            EvalProgressEventKind.RUN_STARTED,
            EvalProgressEventKind.PHASE_STARTED,
            EvalProgressEventKind.PHASE_COMPLETED,
            EvalProgressEventKind.PHASE_STARTED,
            EvalProgressEventKind.PHASE_COMPLETED,
            EvalProgressEventKind.CASE_COMPLETED,
            EvalProgressEventKind.RUN_FINISHED,
        )
        assert _phase_events(reporter) == (
            EvalPhase.CANDIDATE,
            EvalPhase.CANDIDATE,
            EvalPhase.DETERMINISTIC_GRADING,
            EvalPhase.DETERMINISTIC_GRADING,
        )
        case_result = result.case_results[0]
        assert result.duration_seconds is not None
        assert case_result.candidate_duration_seconds is not None
        assert case_result.deterministic_duration_seconds is not None
        assert case_result.judge_duration_seconds is None
        assert case_result.total_duration_seconds is not None

    def test_progress_accounts_for_filtered_repetitions(self) -> None:
        """Report selected-suite totals across candidate repetitions."""
        reporter = _ProgressReporter()
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_case("ba-dev-003"),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    repetitions=2,
                    case_id="ba-dev-003",
                ),
            ),
        )

        case_events = [
            event
            for event in reporter.events
            if event.kind is EvalProgressEventKind.CASE_COMPLETED
        ]
        assert [event.total_cases for event in reporter.events] == [2] * len(
            reporter.events,
        )
        assert [event.completed_cases for event in case_events] == [1, 2]
        assert [event.repetition for event in case_events] == [1, 2]
        assert {event.case_id for event in case_events} == {"ba-dev-003"}

    def test_progress_reports_configured_judge_repetition_count(self) -> None:
        """Display judge repetition settings without adding judge calls."""
        reporter = _ProgressReporter()
        judge = _Judge([_passing_judge_grade()])
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
            judge_service=RubricJudgeService(judge),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    judge_model="judge:local",
                    judge_repetitions=3,
                ),
            ),
        )

        semantic_events = [
            event
            for event in reporter.events
            if event.phase is EvalPhase.SEMANTIC_JUDGING
        ]
        assert judge.calls == 1
        assert result.case_results[0].judge_duration_seconds is not None
        assert {event.judge_repetition for event in semantic_events} == {1}
        assert {
            event.total_judge_repetitions for event in semantic_events
        } == {3}

    def test_progress_skips_judge_for_deterministic_failure(self) -> None:
        """Do not emit judging progress when hard gates fail."""
        reporter = _ProgressReporter()
        judge = _Judge([_passing_judge_grade()])
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(
                _candidate(
                    tool_calls=(
                        ObservedToolCall(
                            name="create_feature",
                            arguments={},
                            status="completed",
                            reached_mcp=True,
                        ),
                    ),
                ),
            ),
            grader=DeterministicEvalGrader(),
            judge_service=RubricJudgeService(judge),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    judge_model="judge:local",
                ),
            ),
        )

        assert judge.calls == 0
        assert result.case_results[0].judge_duration_seconds is None
        assert EvalPhase.SEMANTIC_JUDGING not in _phase_events(reporter)

    def test_progress_marks_expected_error_not_applicable(self) -> None:
        """Skip judging progress for semantic-not-applicable cases."""
        reporter = _ProgressReporter()
        candidate = replace(
            _candidate(tool_calls=()),
            status="failed",
            error_type="AgentSessionBindingError",
            error_message="Agent session is already bound.",
            error_stage=EvalErrorStage.SESSION_BINDING.value,
        )
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(candidate),
            grader=DeterministicEvalGrader(),
            judge_service=RubricJudgeService(_Judge([_passing_judge_grade()])),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_expected_error_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    judge_model="judge:local",
                ),
            ),
        )

        assert result.case_results[0].verdict is EvalVerdict.PASSED
        assert result.case_results[0].judge_duration_seconds is None
        assert EvalPhase.SEMANTIC_JUDGING not in _phase_events(reporter)

    def test_progress_records_candidate_error_timing(self) -> None:
        """Candidate failures still complete candidate and grading phases."""
        reporter = _ProgressReporter()
        runner = EvalRunner(
            candidate_runner=_FailingCandidateRunner(),
            grader=DeterministicEvalGrader(),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                ),
            ),
        )

        assert result.case_results[0].candidate_result.status == "failed"
        assert result.case_results[0].candidate_duration_seconds is not None
        assert reporter.events[-1].kind is EvalProgressEventKind.RUN_FINISHED

    def test_infrastructure_setup_failure_retries_once_then_succeeds(
        self,
    ) -> None:
        """Retry clean pre-execution infrastructure failures once."""
        candidate_runner = _SequenceCandidateRunner(
            (
                _infrastructure_error_candidate(),
                replace(
                    _candidate(tool_calls=()),
                    final_response="First-only requirement.",
                ),
            ),
        )
        runner = EvalRunner(
            candidate_runner=candidate_runner,
            grader=DeterministicEvalGrader(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_case("ba-dev-014"),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=1,
                ),
            ),
        )
        candidate = result.case_results[0].candidate_result

        assert candidate_runner.calls == 2
        assert candidate.status == "completed"
        assert candidate.attempt_count == 2
        assert candidate.retry_count == 1
        assert [attempt.status for attempt in candidate.attempts] == [
            "infrastructure_error",
            "completed",
        ]

    def test_infrastructure_setup_exhausts_retries_without_agent_failure(
        self,
    ) -> None:
        """Persist exhausted setup errors separately from agent quality."""
        candidate_runner = _SequenceCandidateRunner(
            (
                _infrastructure_error_candidate(),
                _infrastructure_error_candidate(),
            ),
        )
        runner = EvalRunner(
            candidate_runner=candidate_runner,
            grader=DeterministicEvalGrader(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=1,
                ),
            ),
        )
        case_result = result.case_results[0]

        assert candidate_runner.calls == 2
        assert case_result.verdict is EvalVerdict.INFRASTRUCTURE_ERROR
        assert case_result.deterministic_grade.hard_gate_failed is False
        assert case_result.deterministic_duration_seconds is None
        assert case_result.judge_grade is None
        assert case_result.candidate_result.retry_count == 1

    def test_ollama_unavailable_without_activity_retries_once(self) -> None:
        """Retry transient provider failures before any observed activity."""
        candidate_runner = _SequenceCandidateRunner(
            (
                _ollama_unavailable_candidate(),
                replace(_candidate(tool_calls=()), final_response="Done."),
            ),
        )
        readiness_calls: list[str] = []
        runner = EvalRunner(
            candidate_runner=candidate_runner,
            grader=DeterministicEvalGrader(),
            infrastructure_readiness_check=lambda: readiness_calls.append(
                "ready",
            ),
            infrastructure_retry_backoff_seconds=0.0,
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=1,
                ),
            ),
        )

        candidate = result.case_results[0].candidate_result
        assert candidate_runner.calls == 2
        assert readiness_calls == ["ready"]
        assert candidate.retry_count == 1
        assert candidate.attempt_count == 2

    def test_read_only_activity_does_not_block_infrastructure_retry(
        self,
    ) -> None:
        """Retry provider failures after only read-only candidate activity."""
        read_only_failure = replace(
            _ollama_unavailable_candidate(),
            skill_calls=(
                ObservedSkillCall(
                    tool_name="load_skill",
                    skill_name="implement-backend-task",
                    status="completed",
                    content_hash="skill-hash",
                ),
            ),
            tool_calls=(
                ObservedToolCall(
                    name="get_feature_overview",
                    arguments={"feature_id": 1},
                    status="completed",
                ),
                ObservedToolCall(
                    name="read_file",
                    arguments={"path": "backend/auth.py"},
                    status="completed",
                ),
            ),
        )
        candidate_runner = _SequenceCandidateRunner(
            (
                read_only_failure,
                replace(_candidate(tool_calls=()), final_response="Done."),
            ),
        )
        runner = EvalRunner(
            candidate_runner=candidate_runner,
            grader=DeterministicEvalGrader(),
            infrastructure_retry_backoff_seconds=0.0,
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=1,
                ),
            ),
        )

        assert candidate_runner.calls == 2
        assert result.case_results[0].candidate_result.retry_count == 1

    def test_infrastructure_retries_zero_disables_retry(self) -> None:
        """Honor explicit zero retry configuration."""
        candidate_runner = _SequenceCandidateRunner(
            (_infrastructure_error_candidate(),),
        )
        runner = EvalRunner(
            candidate_runner=candidate_runner,
            grader=DeterministicEvalGrader(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=0,
                ),
            ),
        )

        assert candidate_runner.calls == 1
        assert result.case_results[0].candidate_result.retry_count == 0

    def test_infrastructure_retry_stops_after_reached_mutations(
        self,
    ) -> None:
        """Do not retry after workspace or workflow mutation delegates."""
        patch_candidate = replace(
            _ollama_unavailable_candidate(),
            tool_calls=(
                ObservedToolCall(
                    name="apply_patch",
                    arguments={"path": "backend/auth.py"},
                    status="completed",
                    reached_mcp=True,
                ),
            ),
        )
        workflow_mutation_candidate = replace(
            _ollama_unavailable_candidate(),
            tool_calls=(
                ObservedToolCall(
                    name="update_task_status",
                    arguments={"task_id": 1, "status": "completed"},
                    status="completed",
                    reached_mcp=True,
                ),
            ),
        )
        effect_candidate = replace(
            _ollama_unavailable_candidate(),
            database_effects=(_database_effect("features", "insert"),),
        )

        for candidate in (
            patch_candidate,
            workflow_mutation_candidate,
            effect_candidate,
        ):
            candidate_runner = _SequenceCandidateRunner((candidate,))
            result = asyncio.run(
                EvalRunner(
                    candidate_runner=candidate_runner,
                    grader=DeterministicEvalGrader(),
                    infrastructure_retry_backoff_seconds=0.0,
                ).run_suite(
                    suite=replace(_suite(), cases=(_first_case(),)),
                    rubric=_rubric(),
                    config=EvalRunConfig(
                        candidate_model="qwen3.5:9b",
                        instructions_hash="instructions-hash",
                        infrastructure_retries=1,
                    ),
                ),
            )

            assert candidate_runner.calls == 1
            assert result.case_results[0].candidate_result.retry_count == 0

    def test_non_infrastructure_errors_are_not_retried(self) -> None:
        """Do not retry candidate-quality or model-selection failures."""
        candidates = (
            _failed_candidate("CapabilityDeniedError"),
            _failed_candidate("MaxTurnsExceeded"),
            _failed_candidate("OllamaModelUnavailableError"),
        )

        for candidate in candidates:
            candidate_runner = _SequenceCandidateRunner((candidate,))
            runner = EvalRunner(
                candidate_runner=candidate_runner,
                grader=DeterministicEvalGrader(),
            )

            asyncio.run(
                runner.run_suite(
                    suite=replace(_suite(), cases=(_first_case(),)),
                    rubric=_rubric(),
                    config=EvalRunConfig(
                        candidate_model="qwen3.5:9b",
                        instructions_hash="instructions-hash",
                        infrastructure_retries=1,
                    ),
                ),
            )

            assert candidate_runner.calls == 1

    def test_ollama_unavailable_after_activity_is_infrastructure_error(
        self,
    ) -> None:
        """Classify exhausted provider outages after read-only activity."""
        candidate = replace(
            _ollama_unavailable_candidate(),
            skill_calls=(
                ObservedSkillCall(
                    tool_name="load_skill",
                    skill_name="design-solution-architecture",
                    status="completed",
                    content_hash="skill-hash",
                ),
            ),
            tool_calls=(
                ObservedToolCall(
                    name="get_feature_overview",
                    arguments={"feature_id": 1},
                    status="completed",
                ),
            ),
        )
        candidate_runner = _SequenceCandidateRunner((candidate, candidate))
        runner = EvalRunner(
            candidate_runner=candidate_runner,
            grader=DeterministicEvalGrader(),
            infrastructure_retry_backoff_seconds=0.0,
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=1,
                ),
            ),
        )

        assert candidate_runner.calls == 2
        assert result.case_results[0].verdict is (
            EvalVerdict.INFRASTRUCTURE_ERROR
        )
        assert result.case_results[0].candidate_result.retry_count == 1
        assert result.case_results[0].deterministic_grade.reasons == (
            "infrastructure_error during candidate_execution: "
            "OllamaUnavailableError",
        )

    def test_configured_retry_count_is_respected(self) -> None:
        """Use no more than the configured infrastructure retry count."""
        candidate_runner = _SequenceCandidateRunner(
            (
                _ollama_unavailable_candidate(),
                _ollama_unavailable_candidate(),
                replace(_candidate(tool_calls=()), final_response="Done."),
            ),
        )
        runner = EvalRunner(
            candidate_runner=candidate_runner,
            grader=DeterministicEvalGrader(),
            infrastructure_retry_backoff_seconds=0.0,
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=2,
                ),
            ),
        )

        candidate = result.case_results[0].candidate_result
        assert candidate_runner.calls == 3
        assert candidate.retry_count == 2
        assert candidate.attempt_count == 3
        assert [attempt.status for attempt in candidate.attempts] == [
            "infrastructure_error",
            "infrastructure_error",
            "completed",
        ]

    def test_progress_reports_infrastructure_retry(self) -> None:
        """Emit a visible bounded retry progress event."""
        reporter = _ProgressReporter()
        runner = EvalRunner(
            candidate_runner=_SequenceCandidateRunner(
                (
                    _infrastructure_error_candidate(),
                    replace(
                        _candidate(tool_calls=()),
                        final_response="First-only requirement.",
                    ),
                ),
            ),
            grader=DeterministicEvalGrader(),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_case("ba-dev-014"),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=1,
                ),
            ),
        )
        retry_events = [
            event
            for event in reporter.events
            if event.kind is EvalProgressEventKind.INFRASTRUCTURE_RETRY
        ]
        candidate = result.case_results[0].candidate_result

        assert len(retry_events) == 1
        assert retry_events[0].case_id == "ba-dev-014"
        assert retry_events[0].infrastructure_retry == 1
        assert retry_events[0].total_infrastructure_retries == 1
        assert result.case_results[0].candidate_duration_seconds is not None
        assert candidate.attempts[0].duration_seconds is not None
        assert result.case_results[0].candidate_duration_seconds >= sum(
            attempt.duration_seconds or 0 for attempt in candidate.attempts
        )

    def test_eta_ignores_exhausted_setup_failure_sample(self) -> None:
        """Do not use failed setup attempts as ETA candidate samples."""
        reporter = _ProgressReporter()
        runner = EvalRunner(
            candidate_runner=_SequenceCandidateRunner(
                (
                    _infrastructure_error_candidate(),
                    replace(
                        _candidate(tool_calls=()),
                        final_response="First-only requirement.",
                    ),
                ),
            ),
            grader=DeterministicEvalGrader(),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        asyncio.run(
            runner.run_suite(
                suite=replace(
                    _suite(),
                    cases=(_first_case(), _case("ba-dev-014")),
                ),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=0,
                ),
            ),
        )

        second_candidate_start = next(
            event
            for event in reporter.events
            if event.kind is EvalProgressEventKind.PHASE_STARTED
            and event.case_id == "ba-dev-014"
            and event.phase is EvalPhase.CANDIDATE
        )
        assert second_candidate_start.estimated_remaining_seconds is None

    def test_progress_records_judge_error_timing(self) -> None:
        """Judge errors still complete semantic judging progress."""
        reporter = _ProgressReporter()
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
            judge_service=RubricJudgeService(_ThrowingJudge()),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    judge_model="judge:local",
                ),
            ),
        )

        assert result.case_results[0].verdict is EvalVerdict.JUDGE_ERROR
        assert result.case_results[0].judge_duration_seconds is not None
        assert EvalPhase.SEMANTIC_JUDGING in _phase_events(reporter)

    def test_progress_reports_cancellation(self) -> None:
        """Report cancellation without swallowing the cancellation."""
        reporter = _ProgressReporter()
        runner = EvalRunner(
            candidate_runner=_CancellingCandidateRunner(),
            grader=DeterministicEvalGrader(),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                runner.run_suite(
                    suite=replace(_suite(), cases=(_first_case(),)),
                    rubric=_rubric(),
                    config=EvalRunConfig(
                        candidate_model="qwen3.5:9b",
                        instructions_hash="instructions-hash",
                    ),
                ),
            )

        assert reporter.events[-1].kind is EvalProgressEventKind.RUN_CANCELLED

    def test_progress_eta_is_unavailable_until_samples_exist(self) -> None:
        """Calculate ETA from measured phase averages."""
        reporter = _ProgressReporter()
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
            progress_reporter=reporter,
            clock=_Clock(),
        )

        asyncio.run(
            runner.run_suite(
                suite=replace(
                    _suite(),
                    cases=(_first_case(), _case("ba-dev-002")),
                ),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                ),
            ),
        )

        candidate_done = _first_event(
            reporter,
            EvalProgressEventKind.PHASE_COMPLETED,
            EvalPhase.CANDIDATE,
        )
        first_case_done = _first_event(
            reporter,
            EvalProgressEventKind.CASE_COMPLETED,
            None,
        )
        assert candidate_done.estimated_remaining_seconds is None
        assert first_case_done.estimated_remaining_seconds is not None

    def test_judge_does_not_override_security_failure(self) -> None:
        """Skip the judge when deterministic hard gates fail."""
        case = _first_case()
        candidate = _candidate(
            tool_calls=(
                ObservedToolCall(
                    name="create_feature",
                    arguments={},
                    status="completed",
                    reached_mcp=True,
                ),
            ),
        )
        judge = _Judge([_passing_judge_grade()])
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(candidate),
            grader=DeterministicEvalGrader(),
            judge_service=RubricJudgeService(judge),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(case,)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    judge_model="judge:local",
                ),
            ),
        )

        assert judge.calls == 0
        assert (
            result.case_results[0].verdict is EvalVerdict.DETERMINISTIC_FAILED
        )

    def test_no_judge_never_calls_judge_and_records_warning(self) -> None:
        """Deterministic-only mode records not_run semantic judging."""
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
            judge_service=None,
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                ),
            ),
        )

        assert result.case_results[0].judge_grade is None
        assert result.case_results[0].verdict is EvalVerdict.NOT_JUDGED
        assert "semantic rubric judge was not run" in result.warnings[0]

    def test_expected_error_boundary_passes_without_judge(self) -> None:
        """Mark deterministic boundary cases as judge not applicable."""
        candidate = replace(
            _candidate(tool_calls=()),
            status="failed",
            error_type="AgentSessionBindingError",
            error_message="Agent session is already bound.",
            error_stage=EvalErrorStage.SESSION_BINDING.value,
        )
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(candidate),
            grader=DeterministicEvalGrader(),
            judge_service=RubricJudgeService(_Judge([_passing_judge_grade()])),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_expected_error_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    judge_model="judge:local",
                ),
            ),
        )

        case_result = result.case_results[0]
        assert case_result.verdict is EvalVerdict.PASSED
        assert case_result.judge_grade is None
        assert case_result.semantic_judge_required is False

    def test_candidate_runner_exception_becomes_case_result(self) -> None:
        """Persist candidate execution failures as deterministic failures."""
        runner = EvalRunner(
            candidate_runner=_FailingCandidateRunner(),
            grader=DeterministicEvalGrader(),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                ),
            ),
        )

        case_result = result.case_results[0]
        assert case_result.candidate_result.status == "failed"
        assert case_result.verdict is EvalVerdict.DETERMINISTIC_FAILED

    def test_invalid_judge_output_retries_once_then_succeeds(self) -> None:
        """Invalid judge output can be corrected once."""
        judge = _Judge(
            [
                JudgeGrade(
                    verdict=EvalVerdict.PASS,
                    scores={},
                    reasons={},
                    confidence=1.0,
                    ambiguous=False,
                    response_preview="{}",
                    raw_response="{}",
                ),
                _passing_judge_grade(),
            ],
        )

        grade = asyncio.run(
            RubricJudgeService(judge).judge_case(
                _first_case(),
                _rubric(),
                _candidate(),
                "judge:local",
            ),
        )

        assert judge.calls == 1
        assert judge.correction_calls == 1
        assert grade.verdict is EvalVerdict.PASS
        assert grade.retry_count == 1

    def test_invalid_judge_repetitions_fail_fast(self) -> None:
        """Reject invalid judge repetition counts."""
        with pytest.raises(ValueError, match="judge_repetitions"):
            asyncio.run(
                RubricJudgeService(_Judge([])).judge_case(
                    _first_case(),
                    _rubric(),
                    _candidate(),
                    "judge:local",
                    judge_repetitions=0,
                ),
            )

    def test_judge_exception_becomes_judge_error(self) -> None:
        """Convert judge adapter exceptions into judge_error grades."""
        grade = asyncio.run(
            RubricJudgeService(_ThrowingJudge()).judge_case(
                _first_case(),
                _rubric(),
                _candidate(),
                "judge:local",
            ),
        )

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert "RuntimeError" in str(grade.error_message)

    def test_correction_exception_becomes_judge_error(self) -> None:
        """Convert correction exceptions into judge_error grades."""
        judge = _ThrowingCorrectionJudge(
            [
                JudgeGrade(
                    verdict=EvalVerdict.PASS,
                    scores={},
                    reasons={},
                    confidence=1.0,
                    ambiguous=False,
                    response_preview="{}",
                    raw_response="{}",
                ),
            ],
        )

        grade = asyncio.run(
            RubricJudgeService(judge).judge_case(
                _first_case(),
                _rubric(),
                _candidate(),
                "judge:local",
            ),
        )

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert grade.retry_count == 1

    def test_two_invalid_judge_outputs_produce_judge_error(self) -> None:
        """Two invalid judge responses never become passes."""
        judge = _Judge(
            [
                JudgeGrade(
                    verdict=EvalVerdict.PASS,
                    scores={},
                    reasons={},
                    confidence=1.0,
                    ambiguous=False,
                    response_preview="{}",
                    raw_response="{}",
                ),
                JudgeGrade(
                    verdict=EvalVerdict.PASS,
                    scores={},
                    reasons={},
                    confidence=1.0,
                    ambiguous=False,
                    response_preview="{}",
                ),
            ],
        )

        grade = asyncio.run(
            RubricJudgeService(judge).judge_case(
                _first_case(),
                _rubric(),
                _candidate(),
                "judge:local",
            ),
        )

        assert judge.calls == 1
        assert judge.correction_calls == 1
        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert grade.retry_count == 1

    def test_unknown_judge_dimension_is_rejected(self) -> None:
        """Reject dimensions that are not in the rubric."""
        grade = _invalid_then_error(
            replace(
                _passing_judge_grade(),
                scores={
                    **_passing_judge_grade().scores,
                    "extra_dimension": 4,
                },
            ),
        )

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert any(
            "unknown dimension" in error for error in grade.validation_errors
        )

    def test_missing_judge_dimension_is_rejected(self) -> None:
        """Reject missing rubric dimensions."""
        passing = _passing_judge_grade()
        bad_scores = dict(passing.scores)
        bad_scores.pop("factual_grounding")

        grade = _invalid_then_error(replace(passing, scores=bad_scores))

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert any(
            "missing dimension" in error for error in grade.validation_errors
        )

    def test_out_of_range_judge_score_is_rejected(self) -> None:
        """Reject judge scores outside the 0 to 4 range."""
        passing = _passing_judge_grade()
        bad_scores = dict(passing.scores)
        bad_scores["factual_grounding"] = 5

        grade = _invalid_then_error(replace(passing, scores=bad_scores))

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert any("0 through 4" in error for error in grade.validation_errors)

    def test_same_model_judging_records_warning(self) -> None:
        """Warn when the candidate and judge models are identical."""
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
            judge_service=RubricJudgeService(_Judge([_passing_judge_grade()])),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(_suite(), cases=(_first_case(),)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    judge_model="qwen3.5:9b",
                ),
            ),
        )

        assert "self-judging bias" in result.warnings[0]

    def test_judge_error_is_persisted_and_later_cases_continue(self) -> None:
        """Case-level judge errors do not abort the evaluation run."""
        judge = _Judge(
            [
                JudgeGrade(
                    verdict=EvalVerdict.JUDGE_ERROR,
                    scores={},
                    reasons={},
                    confidence=0.0,
                    ambiguous=True,
                    error_message="root: invalid JSON",
                    validation_errors=("root: invalid JSON",),
                    response_preview="not-json",
                    raw_response="not-json",
                ),
                JudgeGrade(
                    verdict=EvalVerdict.JUDGE_ERROR,
                    scores={},
                    reasons={},
                    confidence=0.0,
                    ambiguous=True,
                    error_message="root: invalid JSON",
                    validation_errors=("root: invalid JSON",),
                    response_preview="not-json",
                ),
                _passing_judge_grade(),
            ],
        )
        suite = _suite()
        case = _first_case()
        runner = EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
            judge_service=RubricJudgeService(judge),
        )

        result = asyncio.run(
            runner.run_suite(
                suite=replace(suite, cases=(case, case)),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    judge_model="judge:local",
                ),
            ),
        )

        assert result.case_results[0].verdict is EvalVerdict.JUDGE_ERROR
        assert result.case_results[1].verdict is EvalVerdict.PASSED

    def test_result_repository_persists_beneath_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """Save and reload structured eval results locally."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        result = _run_result("run-1")

        repository.save(result)
        loaded = repository.get("run-1")

        assert loaded == result
        assert repository.list_ids() == ["run-1"]
        assert (tmp_path / "evals" / "run-1.json").exists()
        assert repository.get("missing") is None

    def test_result_repository_persists_duration_fields(
        self,
        tmp_path: Path,
    ) -> None:
        """Save and reload run and case durations."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        case_result = replace(
            _run_result("run-1").case_results[0],
            candidate_duration_seconds=61.9,
            deterministic_duration_seconds=1.2,
            judge_duration_seconds=3661.4,
            total_duration_seconds=3724.5,
        )
        result = replace(
            _run_result("run-1"),
            case_results=(case_result,),
            duration_seconds=3724.5,
        )

        repository.save(result)
        loaded = repository.get("run-1")

        assert loaded == result

    def test_result_repository_persists_thinking_configuration(
        self,
        tmp_path: Path,
    ) -> None:
        """Save and reload explicit local-model thinking settings."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        result = replace(
            _run_result("run-1"),
            candidate_thinking_enabled=True,
            judge_thinking_enabled=False,
        )

        repository.save(result)
        loaded = repository.get("run-1")

        assert loaded == result

    def test_result_repository_persists_skill_call_diagnostics(
        self,
        tmp_path: Path,
    ) -> None:
        """Save and reload observed Agent Skill usage diagnostics."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        base = _run_result("run-1")
        candidate = replace(
            base.case_results[0].candidate_result,
            skill_calls=(
                ObservedSkillCall(
                    tool_name="load_skill",
                    skill_name="write-requirements-artifact",
                    status="completed",
                    content_hash="skill-hash",
                ),
            ),
        )
        result = replace(
            base,
            case_results=(
                replace(base.case_results[0], candidate_result=candidate),
            ),
        )

        repository.save(result)
        loaded = repository.get("run-1")

        assert loaded == result

    def test_result_repository_persists_infrastructure_attempts(
        self,
        tmp_path: Path,
    ) -> None:
        """Save and reload bounded infrastructure retry metadata."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        candidate = replace(
            _infrastructure_error_candidate(),
            attempt_count=2,
            retry_count=1,
            attempts=(
                _attempt(1, "infrastructure_error"),
                _attempt(2, "infrastructure_error"),
            ),
        )
        case_result = replace(
            _run_result("run-1").case_results[0],
            candidate_result=candidate,
            verdict=EvalVerdict.INFRASTRUCTURE_ERROR,
        )
        result = replace(
            _run_result("run-1"),
            case_results=(case_result,),
        )

        repository.save(result)
        loaded = repository.get("run-1")

        assert loaded == result

    def test_result_repository_persists_output_token_limit(
        self,
        tmp_path: Path,
    ) -> None:
        """Save and reload per-case candidate output token limits."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        base = _run_result("run-1")
        candidate = replace(
            base.case_results[0].candidate_result,
            max_output_tokens=64,
        )
        result = replace(
            base,
            case_results=(
                replace(base.case_results[0], candidate_result=candidate),
            ),
        )

        repository.save(result)
        loaded = repository.get("run-1")

        assert loaded is not None
        assert loaded.case_results[0].candidate_result.max_output_tokens == 64

    def test_result_repository_loads_historical_durationless_results(
        self,
        tmp_path: Path,
    ) -> None:
        """Keep historical result JSON readable without duration fields."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        repository.save(_run_result("historical"))
        path = tmp_path / "evals" / "historical.json"
        parsed = json.loads(path.read_text())
        parsed.pop("duration_seconds")
        parsed.pop("candidate_thinking_enabled")
        parsed.pop("judge_thinking_enabled")
        for case_result in parsed["case_results"]:
            case_result.pop("intent")
            case_result.pop("context_policy")
            case_result.pop("candidate_duration_seconds")
            case_result.pop("deterministic_duration_seconds")
            case_result.pop("judge_duration_seconds")
            case_result.pop("total_duration_seconds")
            case_result["candidate_result"].pop("skill_calls")
            case_result["candidate_result"].pop("attempt_count")
            case_result["candidate_result"].pop("retry_count")
            case_result["candidate_result"].pop("attempts")
            case_result["candidate_result"].pop("max_output_tokens")
        path.write_text(json.dumps(parsed))

        loaded = repository.get("historical")

        assert loaded is not None
        assert loaded.duration_seconds is None
        assert loaded.candidate_thinking_enabled is False
        assert loaded.judge_thinking_enabled is None
        assert loaded.case_results[0].candidate_duration_seconds is None
        assert loaded.case_results[0].judge_duration_seconds is None
        assert loaded.case_results[0].candidate_result.skill_calls == ()
        assert loaded.case_results[0].candidate_result.attempt_count == 1
        assert loaded.case_results[0].candidate_result.retry_count == 0
        assert loaded.case_results[0].candidate_result.attempts == ()
        assert (
            loaded.case_results[0].candidate_result.max_output_tokens is None
        )
        assert loaded.case_results[0].intent is EvalCaseIntent.UNSPECIFIED
        assert (
            loaded.case_results[0].context_policy
            is EvalContextPolicy.STANDARD_FEATURE_CONTEXT
        )

    def test_result_repository_persists_judge_error_diagnostics(
        self,
        tmp_path: Path,
    ) -> None:
        """Save and reload sanitized judge error metadata."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        judge_error = JudgeGrade(
            verdict=EvalVerdict.JUDGE_ERROR,
            scores={},
            reasons={},
            confidence=0.0,
            ambiguous=True,
            error_message="root: invalid JSON",
            judge_model="qwen3.8:27b",
            response_hash="response-hash",
            response_preview="not-json",
            validation_errors=("root: invalid JSON",),
            retry_count=1,
        )
        case_result = replace(
            _run_result("run-1").case_results[0],
            judge_grade=judge_error,
            verdict=EvalVerdict.JUDGE_ERROR,
        )
        result = replace(
            _run_result("run-1"),
            judge_model="qwen3.8:27b",
            case_results=(case_result,),
        )

        repository.save(result)
        loaded = repository.get("run-1")

        assert loaded is not None
        loaded_grade = loaded.case_results[0].judge_grade
        assert loaded_grade is not None
        assert loaded_grade.verdict is EvalVerdict.JUDGE_ERROR
        assert loaded_grade.response_preview == "not-json"
        assert loaded_grade.validation_errors == ("root: invalid JSON",)
        assert loaded_grade.retry_count == 1

    def test_result_repository_rejects_invalid_json(
        self,
        tmp_path: Path,
    ) -> None:
        """Fail closed when a saved result is not a JSON object."""
        repository = JsonEvalResultRepository(tmp_path / "evals")
        (tmp_path / "evals" / "bad.json").write_text("[]")

        with pytest.raises(ValueError, match="object"):
            repository.get("bad")

    def test_comparison_rejects_mismatched_hashes(self) -> None:
        """Refuse non-equivalent comparisons by default."""
        baseline = _run_result("baseline")
        candidate = replace(_run_result("candidate"), dataset_hash="other")

        with pytest.raises(ValueError, match="Dataset hashes differ"):
            EvalComparisonService().compare(baseline, candidate)

    def test_comparison_detects_regressions(self) -> None:
        """Report pairwise regressions explicitly."""
        baseline = _run_result("baseline", judge_grade=_passing_judge_grade())
        candidate_case = replace(
            baseline.case_results[0],
            verdict=EvalVerdict.FAIL,
        )
        candidate = replace(
            baseline,
            id="candidate",
            case_results=(candidate_case,),
        )

        comparison = EvalComparisonService().compare(baseline, candidate)

        assert comparison.regressed_cases == ("ba-dev-001",)
        assert comparison.improved_cases == ()

    def test_missing_judge_grade_is_semantically_uncomparable(self) -> None:
        """Never infer semantic changes without judge grades on both sides."""
        baseline = _run_result("baseline")
        candidate_case = replace(
            baseline.case_results[0],
            judge_grade=_passing_judge_grade(),
            verdict=EvalVerdict.PASSED,
        )
        candidate = replace(
            baseline,
            id="candidate",
            judge_model="qwen3.8:27b",
            case_results=(candidate_case,),
        )

        comparison = EvalComparisonService().compare(baseline, candidate)

        assert comparison.semantic_improved_cases == ()
        assert comparison.semantic_uncomparable_cases == ("ba-dev-001",)

    def test_deterministic_failure_without_judge_is_not_semantic(self) -> None:
        """Do not classify unjudged deterministic failures semantically."""
        baseline_case = replace(
            _run_result("baseline").case_results[0],
            deterministic_grade=DeterministicGrade(
                passed=False,
                hard_gate_failed=True,
                reasons=("candidate failed OllamaUnavailableError",),
            ),
            judge_grade=None,
            verdict=EvalVerdict.DETERMINISTIC_FAILED,
        )
        baseline = replace(
            _run_result("baseline"),
            case_results=(baseline_case,),
        )
        candidate_case = replace(
            baseline_case,
            deterministic_grade=DeterministicGrade(
                passed=True,
                hard_gate_failed=False,
                reasons=(),
            ),
            judge_grade=_passing_judge_grade(),
            verdict=EvalVerdict.PASSED,
        )
        candidate = replace(
            baseline,
            id="candidate",
            judge_model="qwen3.8:27b",
            case_results=(candidate_case,),
        )

        comparison = EvalComparisonService().compare(baseline, candidate)

        assert comparison.deterministic_improved_cases == ("ba-dev-001",)
        assert comparison.semantic_improved_cases == ()
        assert comparison.semantic_uncomparable_cases == ("ba-dev-001",)

    def test_infrastructure_errors_are_not_comparable(self) -> None:
        """Do not classify infrastructure failures as regressions."""
        baseline = _run_result("baseline", judge_grade=_passing_judge_grade())
        candidate_case = replace(
            baseline.case_results[0],
            candidate_result=_infrastructure_error_candidate(),
            deterministic_grade=DeterministicGrade(
                passed=False,
                hard_gate_failed=False,
                reasons=("infrastructure_error before candidate execution",),
            ),
            judge_grade=None,
            verdict=EvalVerdict.INFRASTRUCTURE_ERROR,
        )
        candidate = replace(
            baseline,
            id="candidate",
            case_results=(candidate_case,),
        )

        comparison = EvalComparisonService().compare(baseline, candidate)

        assert comparison.regressed_cases == ()
        assert comparison.deterministic_regressed_cases == ()
        assert comparison.deterministic_uncomparable_cases == ("ba-dev-001",)
        assert comparison.semantic_uncomparable_cases == ("ba-dev-001",)

    def test_judged_results_remain_semantically_comparable(self) -> None:
        """Compare semantic verdicts when both sides have judge grades."""
        failing_grade = replace(
            _passing_judge_grade(),
            verdict=EvalVerdict.FAIL,
        )
        baseline = _run_result("baseline", judge_grade=failing_grade)
        candidate_case = replace(
            baseline.case_results[0],
            judge_grade=_passing_judge_grade(),
            verdict=EvalVerdict.PASSED,
        )
        candidate = replace(
            baseline,
            id="candidate",
            case_results=(candidate_case,),
        )

        comparison = EvalComparisonService().compare(baseline, candidate)

        assert comparison.semantic_improved_cases == ("ba-dev-001",)

    def test_calibration_metrics_are_calculated(self) -> None:
        """Compare judge grades with human labels."""
        result = _run_result("run-1", judge_grade=_passing_judge_grade())
        labels = (
            HumanLabel(
                case_id="ba-dev-001",
                rubric_id="business_analyst_workflow",
                rubric_version="2026-08-20",
                scores={dimension.id: 4 for dimension in _rubric().dimensions},
                verdict=EvalVerdict.PASS,
                reason="Matches expected behavior.",
                rater="human",
                rated_at="2026-08-20T00:00:00Z",
            ),
        )

        calibration = JudgeCalibrationService().calibrate(result, labels)

        assert calibration.verdict_agreement == 1.0
        assert calibration.judge_ambiguity_rate == 0.0
        assert calibration.disagreements == ()


def _suite() -> EvalSuite:
    return JsonlGoldenDatasetLoader().load(
        "business_analyst_development",
        Path("evals/datasets/business_analyst_development.jsonl"),
    )


def _architect_suite() -> EvalSuite:
    return JsonlGoldenDatasetLoader().load(
        "software_architect_development",
        Path("evals/datasets/software_architect_development.jsonl"),
    )


def _first_case() -> EvalCase:
    return _suite().cases[0]


def _case(case_id: str) -> EvalCase:
    for case in _suite().cases:
        if case.id == case_id:
            return case
    raise AssertionError(f"Missing test case {case_id}.")


def _case_from_suite(suite: EvalSuite, case_id: str) -> EvalCase:
    for case in suite.cases:
        if case.id == case_id:
            return case
    raise AssertionError(f"Missing test case {case_id}.")


def _tool_names(case: EvalCase) -> set[str]:
    return {tool.name for tool in case.expected_tool_calls}


def _find_symbol_name(case: EvalCase) -> str:
    for tool in case.expected_tool_calls:
        if tool.name == "find_symbol":
            name = tool.arguments_subset.get("name")
            if isinstance(name, str):
                return name
    raise AssertionError(f"Missing exact find_symbol name for {case.id}.")


def _expected_status_update(case: EvalCase) -> dict[str, object]:
    for effect in case.expected_database_effects:
        if (
            effect.table == "development_tasks"
            and effect.operation == "update"
        ):
            return effect.field_values
    raise AssertionError(f"Missing expected task update for {case.id}.")


def _minimal_case_record() -> dict[str, object]:
    return {
        "id": "legacy-case",
        "name": "Legacy case",
        "category": "legacy",
        "severity": "medium",
        "active_role": "business_analyst",
        "feature_fixtures": [],
        "prior_session_turns": [],
        "user_input": "List the current features.",
        "expected_tool_calls": [
            {
                "name": "list_features",
                "arguments_subset": {},
            },
        ],
        "forbidden_tool_calls": [],
        "expected_database_effects": [],
        "forbidden_database_effects": [],
        "required_response_facts": [],
        "forbidden_response_claims": [],
        "rubric_id": "business_analyst_workflow",
        "note": "Historical JSONL compatibility case.",
    }


def _trajectory(tool_name: str, feature_id: int) -> ExpectedToolTrajectory:
    return ExpectedToolTrajectory(
        required_tool_calls=(
            ExpectedToolCall(
                name=tool_name,
                arguments_subset={"feature_id": feature_id},
            ),
        ),
    )


def _event_kinds(
    reporter: _ProgressReporter,
) -> tuple[EvalProgressEventKind, ...]:
    return tuple(event.kind for event in reporter.events)


def _phase_events(
    reporter: _ProgressReporter,
) -> tuple[EvalPhase, ...]:
    return tuple(
        event.phase for event in reporter.events if event.phase is not None
    )


def _first_event(
    reporter: _ProgressReporter,
    kind: EvalProgressEventKind,
    phase: EvalPhase | None,
) -> EvalProgressEvent:
    for event in reporter.events:
        if event.kind is kind and event.phase is phase:
            return event
    raise AssertionError(f"Missing progress event {kind} {phase}.")


def _expected_error_case() -> EvalCase:
    return replace(
        _first_case(),
        expected_tool_calls=(),
        forbidden_tool_calls=("create_feature",),
        objective_response_facts=(),
        prohibited_objective_claims=(),
        expected_error=ExpectedError(
            error_type="AgentSessionBindingError",
            stage=EvalErrorStage.SESSION_BINDING,
            message_fragment="already bound",
        ),
        semantic_judge_required=False,
    )


def _rubric() -> Rubric:
    return MarkdownRubricLoader().load(
        Path("evals/rubrics/business_analyst.md"),
    )


def _candidate(
    tool_calls: tuple[ObservedToolCall, ...] = (
        ObservedToolCall(
            name="list_features",
            arguments={},
            status="completed",
        ),
    ),
) -> CandidateRunResult:
    return CandidateRunResult(
        role=DevelopmentRole.BUSINESS_ANALYST,
        model="qwen3.5:9b",
        final_response="Login",
        tool_calls=tool_calls,
        database_effects=(),
    )


def _infrastructure_error_candidate() -> CandidateRunResult:
    return CandidateRunResult(
        role=DevelopmentRole.BUSINESS_ANALYST,
        model="qwen3.5:9b",
        final_response="",
        tool_calls=(),
        database_effects=(),
        status=EvalVerdict.INFRASTRUCTURE_ERROR.value,
        error_type="WorkflowMCPUnavailableError",
        error_message="Development workflow MCP server could not start.",
        error_stage=EvalErrorStage.INFRASTRUCTURE_SETUP.value,
    )


def _ollama_unavailable_candidate() -> CandidateRunResult:
    return CandidateRunResult(
        role=DevelopmentRole.BUSINESS_ANALYST,
        model="qwen3.5:9b",
        final_response="",
        tool_calls=(),
        database_effects=(),
        status=EvalVerdict.INFRASTRUCTURE_ERROR.value,
        error_type="OllamaUnavailableError",
        error_message="Local Ollama unavailable.",
        error_stage=EvalErrorStage.CANDIDATE_EXECUTION.value,
    )


def _failed_candidate(error_type: str) -> CandidateRunResult:
    return CandidateRunResult(
        role=DevelopmentRole.BUSINESS_ANALYST,
        model="qwen3.5:9b",
        final_response="",
        tool_calls=(),
        database_effects=(),
        status="failed",
        error_type=error_type,
        error_message=f"{error_type} occurred.",
        error_stage=EvalErrorStage.CANDIDATE_EXECUTION.value,
    )


def _database_effect(table: str, operation: str) -> DatabaseEffect:
    return DatabaseEffect(
        table=table,
        operation=operation,
        field_values={"id": 1},
    )


def _attempt(attempt: int, status: str) -> EvalAttemptResult:
    return EvalAttemptResult(
        attempt=attempt,
        status=status,
        duration_seconds=0.1,
        error_type="WorkflowMCPUnavailableError",
        error_stage=EvalErrorStage.INFRASTRUCTURE_SETUP.value,
    )


def _passing_judge_grade_for_case(case_id: str) -> JudgeGrade:
    rubric = _rubric()
    return JudgeGrade(
        verdict=EvalVerdict.PASS,
        scores={dimension.id: 4 for dimension in rubric.dimensions},
        reasons={
            dimension.id: "Observable output is acceptable."
            for dimension in rubric.dimensions
        },
        confidence=0.9,
        ambiguous=False,
        rubric_id=rubric.id,
        rubric_version=rubric.version,
        case_id=case_id,
        evidence={
            dimension.id: "Observed output supports the score."
            for dimension in rubric.dimensions
        },
        judge_model="judge:local",
    )


def _passing_judge_grade(case_id: str = "ba-dev-001") -> JudgeGrade:
    return _passing_judge_grade_for_case(case_id)


def _invalid_then_error(invalid_grade: JudgeGrade) -> JudgeGrade:
    judge = _Judge(
        [
            replace(
                invalid_grade,
                raw_response="{}",
                response_preview="{}",
            ),
            JudgeGrade(
                verdict=EvalVerdict.JUDGE_ERROR,
                scores={},
                reasons={},
                confidence=0.0,
                ambiguous=True,
                error_message="root: invalid JSON",
                validation_errors=("root: invalid JSON",),
                response_preview="{}",
            ),
        ],
    )
    return asyncio.run(
        RubricJudgeService(judge).judge_case(
            _first_case(),
            _rubric(),
            _candidate(),
            "judge:local",
        ),
    )


def _run_result(
    result_id: str,
    judge_grade: JudgeGrade | None = None,
) -> EvalRunResult:
    suite = _suite()
    case_result = asyncio.run(
        EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
        ).run_suite(
            suite=replace(suite, cases=(suite.cases[0],)),
            rubric=_rubric(),
            config=EvalRunConfig(
                candidate_model="qwen3.5:9b",
                instructions_hash="instructions-hash",
            ),
        ),
    ).case_results[0]
    if judge_grade is not None:
        case_result = replace(
            case_result,
            judge_grade=judge_grade,
            verdict=judge_grade.verdict,
        )
    result = asyncio.run(
        EvalRunner(
            candidate_runner=_CandidateRunner(_candidate()),
            grader=DeterministicEvalGrader(),
        ).run_suite(
            suite=replace(suite, cases=(suite.cases[0],)),
            rubric=_rubric(),
            config=EvalRunConfig(
                candidate_model="qwen3.5:9b",
                instructions_hash="instructions-hash",
            ),
        ),
    )
    return replace(result, id=result_id, case_results=(case_result,))
