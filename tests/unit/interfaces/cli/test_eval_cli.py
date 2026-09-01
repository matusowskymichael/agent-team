"""Tests for the human-only evaluation CLI."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.deterministic_grade import DeterministicGrade
from agent_team.domain.evaluation.eval_attempt_result import (
    EvalAttemptResult,
)
from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_run_config import EvalRunConfig
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.observed_skill_call import ObservedSkillCall
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.interfaces.cli import eval_cli


class _Repository:
    def __init__(self) -> None:
        self.results = {
            "baseline": _result("baseline", EvalVerdict.FAIL),
            "candidate": _result("candidate", EvalVerdict.NOT_RUN),
        }
        self.saved: EvalRunResult | None = None

    def save(self, result: EvalRunResult) -> None:
        self.saved = result
        self.results[result.id] = result

    def get(self, result_id: str) -> EvalRunResult | None:
        return self.results.get(result_id)

    def list_ids(self) -> list[str]:
        return sorted(self.results)


class _EvalRunner:
    received_case_ids: tuple[str, ...] = ()
    received_progress_reporter: object | None = None
    received_infrastructure_retries: int | None = None
    received_rubric_id: str | None = None

    def __init__(self, **_kwargs: object) -> None:
        self.__class__.received_progress_reporter = _kwargs.get(
            "progress_reporter",
        )

    async def run_suite(
        self,
        suite: EvalSuite,
        **_kwargs: object,
    ) -> EvalRunResult:
        config = _kwargs.get("config")
        rubric = _kwargs.get("rubric")
        assert isinstance(config, EvalRunConfig)
        assert isinstance(rubric, Rubric)
        self.__class__.received_infrastructure_retries = (
            config.infrastructure_retries
        )
        self.__class__.received_rubric_id = rubric.id
        self.__class__.received_case_ids = tuple(
            case.id for case in suite.cases
        )
        return _result("eval-run-1", EvalVerdict.NOT_JUDGED)


class _InfrastructureEvalRunner:
    def __init__(self, **_kwargs: object) -> None:
        pass

    async def run_suite(
        self,
        suite: EvalSuite,
        **_kwargs: object,
    ) -> EvalRunResult:
        assert suite.cases
        return _infrastructure_result()


class TestEvalCli:
    """Evaluation CLI behavior tests."""

    def test_list_suites_prints_golden_datasets(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """List manually maintained local dataset suites."""
        exit_code = eval_cli.main(["list-suites"])

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "business_analyst_development" in captured.out
        assert "business_analyst_holdout" in captured.out
        assert "backend_developer_development" in captured.out
        assert "frontend_developer_development" in captured.out
        assert "software_architect_development" in captured.out
        assert "software_architect_holdout" in captured.out
        assert captured.err == ""

    def test_run_requires_judge_model_unless_no_judge(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Return a concise configuration error without running models."""
        monkeypatch.delenv(eval_cli.OLLAMA_JUDGE_MODEL_ENV, raising=False)

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--candidate-model",
                "qwen3.5:9b",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 2
        assert "Judge model is required" in captured.err

    def test_run_no_judge_saves_result_with_fake_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Run deterministic-only eval CLI path without contacting Ollama."""
        repository = _Repository()

        def ensure_model_ready(_settings: object) -> None:
            return None

        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli, "EvalRunner", _EvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--candidate-model",
                "qwen3.5:9b",
                "--no-judge",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert repository.saved is not None
        assert "Eval run: eval-run-1" in captured.out
        assert "Deterministic:" in captured.out
        assert "Semantic judge:" in captured.out
        assert "not run: 1" in captured.out
        assert "not applicable: 0" in captured.out
        assert "Infrastructure:" in captured.out
        assert "  retries: 0" in captured.out
        assert "  failures: 0" in captured.out
        assert "Overall fully evaluated:" in captured.out
        assert "  passed: 0" in captured.out
        assert _EvalRunner.received_infrastructure_retries == 1

    def test_run_software_architect_suite_uses_architect_rubric(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Select the architect rubric for architect datasets."""
        repository = _Repository()

        def ensure_model_ready(_settings: object) -> None:
            return None

        _EvalRunner.received_rubric_id = None
        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli, "EvalRunner", _EvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "software_architect_development",
                "--candidate-model",
                "qwen3.5:9b",
                "--no-judge",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert _EvalRunner.received_rubric_id == "software_architect_workflow"
        assert len(_EvalRunner.received_case_ids) == 24
        assert captured.err == ""

    def test_run_backend_developer_suite_uses_backend_rubric(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Select the backend rubric for backend developer datasets."""
        repository = _Repository()

        def ensure_model_ready(_settings: object) -> None:
            return None

        _EvalRunner.received_rubric_id = None
        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli, "EvalRunner", _EvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "backend_developer_development",
                "--candidate-model",
                "qwen3.5:9b",
                "--no-judge",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert _EvalRunner.received_rubric_id == "backend_developer_workflow"
        assert len(_EvalRunner.received_case_ids) == 10
        assert captured.err == ""

    def test_run_frontend_developer_suite_uses_frontend_rubric(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Select the frontend rubric for frontend developer datasets."""
        repository = _Repository()

        def ensure_model_ready(_settings: object) -> None:
            return None

        _EvalRunner.received_rubric_id = None
        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli, "EvalRunner", _EvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "frontend_developer_development",
                "--candidate-model",
                "qwen3.5:9b",
                "--no-judge",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert _EvalRunner.received_rubric_id == "frontend_developer_workflow"
        assert len(_EvalRunner.received_case_ids) == 10
        assert captured.err == ""

    def test_run_returns_system_error_for_infrastructure_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Do not report exhausted infrastructure failures as quality fails."""
        repository = _Repository()

        def ensure_model_ready(_settings: object) -> None:
            return None

        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli, "EvalRunner", _InfrastructureEvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--candidate-model",
                "qwen3.5:9b",
                "--no-judge",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == eval_cli.EXIT_SYSTEM_ERROR
        assert "Infrastructure:" in captured.out
        assert "  failures: 1" in captured.out
        assert "Deterministic:" in captured.out
        assert "  failed: 0" in captured.out

    def test_run_case_id_filters_canonical_suite(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Run only the selected case without changing dataset hashes."""
        repository = _Repository()

        def ensure_model_ready(_settings: object) -> None:
            return None

        _EvalRunner.received_case_ids = ()
        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli, "EvalRunner", _EvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--case-id",
                "ba-dev-003",
                "--candidate-model",
                "qwen3.5:9b",
                "--no-judge",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert _EvalRunner.received_case_ids == ("ba-dev-003",)
        assert "Case filter:" in captured.out

    def test_run_accepts_zero_infrastructure_retries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Allow disabling bounded infrastructure retries explicitly."""
        repository = _Repository()

        def ensure_model_ready(_settings: object) -> None:
            return None

        _EvalRunner.received_infrastructure_retries = None
        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli, "EvalRunner", _EvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--candidate-model",
                "qwen3.5:9b",
                "--infrastructure-retries",
                "0",
                "--no-judge",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.err == ""
        assert _EvalRunner.received_infrastructure_retries == 0

    def test_unknown_case_id_fails_before_model_check(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Reject unknown selected cases before checking local models."""
        model_checks = 0

        def ensure_model_ready(_settings: object) -> None:
            nonlocal model_checks
            model_checks += 1

        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--case-id",
                "missing-case",
                "--candidate-model",
                "qwen3.5:9b",
                "--no-judge",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 2
        assert model_checks == 0
        assert "missing-case" in captured.err

    def test_show_compare_and_calibrate_read_saved_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Exercise read-only eval result commands."""
        repository = _Repository()
        labels_path = tmp_path / "labels.jsonl"
        labels_path.write_text(
            (
                '{"case_id":"ba-dev-001","rubric_id":"business_analyst_'
                'workflow","rubric_version":"2026-08-20","scores":{},'
                '"verdict":"pass","reason":"ok","rater":"human",'
                '"rated_at":"2026-08-20T00:00:00Z"}\n'
            ),
        )
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        show_code = eval_cli.main(["show", "baseline"])
        compare_code = eval_cli.main(["compare", "baseline", "candidate"])
        calibrate_code = eval_cli.main(
            [
                "calibrate",
                "--eval-run-id",
                "baseline",
                "--human-labels",
                str(labels_path),
            ],
        )

        captured = capsys.readouterr()

        assert show_code == 0
        assert compare_code == 0
        assert calibrate_code == 0
        assert "Case: ba-dev-001" in captured.out
        assert "Duration: -" in captured.out
        assert "Expected tool calls:" in captured.out
        assert "Observed database effects:" in captured.out
        assert "Candidate final response:" not in captured.out
        assert "Deterministic improvements:" in captured.out
        assert "Semantic not comparable:" in captured.out
        assert "Verdict agreement:" in captured.out

    def test_show_displays_judge_error_details(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Display persisted judge status, scores, and validation errors."""
        repository = _Repository()
        repository.results["judged"] = _judged_error_result()
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(["show", "judged"])

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Results:" in captured.out
        assert "  judge_error: 1" in captured.out
        assert "Judge validation errors: root: invalid JSON" in captured.out
        assert "Judge output preview: not-json" in captured.out

    def test_show_separates_infrastructure_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Display infrastructure errors outside deterministic failures."""
        repository = _Repository()
        repository.results["infra"] = _infrastructure_result()
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(["show", "infra"])

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Infrastructure:" in captured.out
        assert "  retries: 1" in captured.out
        assert "  failures: 1" in captured.out
        assert "Overall:" in captured.out
        assert "  infrastructure_error: 1" in captured.out
        assert "Deterministic checks: not run" in captured.out

    def test_show_verbose_displays_sanitized_diagnostics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verbose show includes detailed sanitized diagnostics."""
        repository = _Repository()
        repository.results["verbose"] = _verbose_result()
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(["show", "verbose", "--verbose"])

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Candidate final response:" in captured.out
        assert "password: [REDACTED]" in captured.out
        assert "swordfish" not in captured.out
        assert "Expected tool trajectories:" in captured.out
        assert "Judge reasons:" in captured.out
        assert "Judge evidence:" in captured.out
        assert "Critical thresholds:" in captured.out
        assert "Final status reason:" in captured.out
        assert "Durations:" in captured.out
        assert "candidate=00:01:01" in captured.out
        assert "Case intent:" in captured.out
        assert "Evaluation context policy:" in captured.out
        assert "Observed skill calls:" in captured.out
        assert "write-requirements-artifact" in captured.out
        assert "skill-hash" in captured.out

    def test_show_unknown_eval_run_returns_concise_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Return non-zero when a saved eval run is missing."""

        def repository_factory() -> _Repository:
            return _Repository()

        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            repository_factory,
        )

        exit_code = eval_cli.main(["show", "missing"])

        captured = capsys.readouterr()

        assert exit_code == 2
        assert "Evaluation run missing was not found." in captured.err

    def test_run_with_judge_checks_only_local_ollama_models(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Check candidate and judge through local Ollama settings."""
        checked_models: list[str] = []
        repository = _Repository()

        def ensure_model_ready(settings: object) -> None:
            assert isinstance(settings, eval_cli.OllamaSettings)
            checked_models.append(settings.model)

        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli, "EvalRunner", _EvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--case-id",
                "ba-dev-001",
                "--candidate-model",
                "qwen3.5:9b",
                "--judge-model",
                "qwen3.8:27b",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert checked_models == ["qwen3.5:9b", "qwen3.8:27b"]
        assert captured.err == ""

    def test_show_displays_stored_duration(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Display stored total duration for saved eval runs."""
        repository = _Repository()
        repository.results["timed"] = _timed_result()
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(["show", "timed"])

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Duration: 01:02:03" in captured.out

    def test_run_no_progress_does_not_create_reporter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Honor --no-progress even if stderr is interactive."""
        repository = _Repository()
        _EvalRunner.received_progress_reporter = object()

        def ensure_model_ready(_settings: object) -> None:
            return None

        monkeypatch.setattr(
            eval_cli,
            "ensure_ollama_model_ready",
            ensure_model_ready,
        )
        monkeypatch.setattr(eval_cli.sys.stderr, "isatty", lambda: True)
        monkeypatch.setattr(eval_cli, "EvalRunner", _EvalRunner)
        monkeypatch.setattr(
            eval_cli,
            "JsonEvalResultRepository",
            lambda: repository,
        )

        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--candidate-model",
                "qwen3.5:9b",
                "--no-judge",
                "--no-progress",
            ],
        )

        captured = capsys.readouterr()

        assert exit_code == 0
        assert _EvalRunner.received_progress_reporter is None
        assert captured.err == ""


def _result(result_id: str, verdict: EvalVerdict) -> EvalRunResult:
    case_result = EvalCaseResult(
        case_id="ba-dev-001",
        repetition=1,
        candidate_result=CandidateRunResult(
            role=DevelopmentRole.BUSINESS_ANALYST,
            model="qwen3.5:9b",
            final_response="Login",
            tool_calls=(),
            database_effects=(),
        ),
        deterministic_grade=DeterministicGrade(
            passed=verdict
            not in {EvalVerdict.FAIL, EvalVerdict.DETERMINISTIC_FAILED},
            hard_gate_failed=verdict
            in {EvalVerdict.FAIL, EvalVerdict.DETERMINISTIC_FAILED},
            reasons=(),
        ),
        judge_grade=None,
        verdict=verdict,
    )
    return EvalRunResult(
        id=result_id,
        suite_id="business_analyst_development",
        candidate_model="qwen3.5:9b",
        judge_model=None,
        dataset_hash="dataset",
        rubric_hash="rubric",
        instructions_hash="instructions",
        package_version="0.1.0",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ended_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        case_results=(case_result,),
        warnings=(),
        case_filter=None,
    )


def _judged_error_result() -> EvalRunResult:
    result = _result("judged", EvalVerdict.JUDGE_ERROR)
    judge_grade = JudgeGrade(
        verdict=EvalVerdict.JUDGE_ERROR,
        scores={},
        reasons={},
        confidence=0.0,
        ambiguous=True,
        error_message="root: invalid JSON",
        response_preview="not-json",
        validation_errors=("root: invalid JSON",),
    )
    return EvalRunResult(
        id=result.id,
        suite_id=result.suite_id,
        candidate_model=result.candidate_model,
        judge_model="qwen3.8:27b",
        dataset_hash=result.dataset_hash,
        rubric_hash=result.rubric_hash,
        instructions_hash=result.instructions_hash,
        package_version=result.package_version,
        started_at=result.started_at,
        ended_at=result.ended_at,
        case_results=(
            EvalCaseResult(
                case_id="ba-dev-001",
                repetition=1,
                candidate_result=result.case_results[0].candidate_result,
                deterministic_grade=DeterministicGrade(
                    passed=True,
                    hard_gate_failed=False,
                    reasons=(),
                ),
                judge_grade=judge_grade,
                verdict=EvalVerdict.JUDGE_ERROR,
            ),
        ),
        warnings=(),
    )


def _infrastructure_result() -> EvalRunResult:
    result = _result("infra", EvalVerdict.INFRASTRUCTURE_ERROR)
    candidate = CandidateRunResult(
        role=DevelopmentRole.BUSINESS_ANALYST,
        model="qwen3.5:9b",
        final_response="",
        tool_calls=(),
        database_effects=(),
        status=EvalVerdict.INFRASTRUCTURE_ERROR.value,
        error_type="WorkflowMCPUnavailableError",
        error_message="Development workflow MCP server could not start.",
        error_stage="infrastructure_setup",
        attempt_count=2,
        retry_count=1,
        attempts=(
            EvalAttemptResult(
                attempt=1,
                status=EvalVerdict.INFRASTRUCTURE_ERROR.value,
                duration_seconds=0.1,
                error_type="WorkflowMCPUnavailableError",
                error_stage="infrastructure_setup",
            ),
            EvalAttemptResult(
                attempt=2,
                status=EvalVerdict.INFRASTRUCTURE_ERROR.value,
                duration_seconds=0.1,
                error_type="WorkflowMCPUnavailableError",
                error_stage="infrastructure_setup",
            ),
        ),
    )
    return replace(
        result,
        case_results=(
            replace(
                result.case_results[0],
                candidate_result=candidate,
                deterministic_grade=DeterministicGrade(
                    passed=False,
                    hard_gate_failed=False,
                    reasons=("infrastructure_error before execution",),
                ),
                verdict=EvalVerdict.INFRASTRUCTURE_ERROR,
            ),
        ),
    )


def _verbose_result() -> EvalRunResult:
    result = _result("verbose", EvalVerdict.PASSED)
    judge_grade = JudgeGrade(
        verdict=EvalVerdict.PASS,
        scores={
            "factual_grounding": 4,
            "completeness": 3,
            "tool_accuracy": 4,
            "role_adherence": 4,
            "least_privilege": 4,
            "clarity": 3,
            "uncertainty": 3,
        },
        reasons={
            "factual_grounding": "Grounded in tool output.",
            "completeness": "Complete enough.",
            "tool_accuracy": "Correct tool.",
            "role_adherence": "Role followed.",
            "least_privilege": "Read-only.",
            "clarity": "Clear.",
            "uncertainty": "Appropriate.",
        },
        confidence=0.9,
        ambiguous=False,
        evidence={
            "factual_grounding": "Observed Login in response.",
            "completeness": "Response includes requested feature.",
            "tool_accuracy": "Observed list_features.",
            "role_adherence": "No role escalation.",
            "least_privilege": "No mutations.",
            "clarity": "Readable.",
            "uncertainty": "No unsupported certainty.",
        },
    )
    case_result = EvalCaseResult(
        case_id="ba-dev-001",
        repetition=1,
        candidate_result=CandidateRunResult(
            role=DevelopmentRole.BUSINESS_ANALYST,
            model="qwen3.5:9b",
            final_response="Login\npassword: swordfish",
            tool_calls=(),
            skill_calls=(
                ObservedSkillCall(
                    tool_name="load_skill",
                    skill_name="write-requirements-artifact",
                    status="completed",
                    content_hash="skill-hash",
                ),
            ),
            database_effects=(),
        ),
        deterministic_grade=DeterministicGrade(
            passed=True,
            hard_gate_failed=False,
            reasons=(),
        ),
        judge_grade=judge_grade,
        verdict=EvalVerdict.PASSED,
        candidate_duration_seconds=61.9,
        deterministic_duration_seconds=1.2,
        judge_duration_seconds=2.8,
        total_duration_seconds=65.9,
    )
    return EvalRunResult(
        id=result.id,
        suite_id=result.suite_id,
        candidate_model=result.candidate_model,
        judge_model="qwen3.8:27b",
        dataset_hash=result.dataset_hash,
        rubric_hash=result.rubric_hash,
        instructions_hash=result.instructions_hash,
        package_version=result.package_version,
        started_at=result.started_at,
        ended_at=result.ended_at,
        case_results=(case_result,),
        warnings=(),
    )


def _timed_result() -> EvalRunResult:
    case_result = replace(
        _result("timed", EvalVerdict.PASSED).case_results[0],
        candidate_duration_seconds=61.9,
        deterministic_duration_seconds=1.2,
        judge_duration_seconds=None,
        total_duration_seconds=63.1,
    )
    return replace(
        _result("timed", EvalVerdict.PASSED),
        case_results=(case_result,),
        duration_seconds=3723.9,
    )
