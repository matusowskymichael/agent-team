"""Fixtures for offline developer golden lifecycle regressions."""

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.deterministic_grade import DeterministicGrade
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.evaluation import local_candidate_agent_runner
from agent_team.infrastructure.evaluation.json_eval_result_repository import (
    JsonEvalResultRepository,
)
from agent_team.infrastructure.evaluation.jsonl_golden_dataset_loader import (
    JsonlGoldenDatasetLoader,
)
from agent_team.infrastructure.evaluation.local_candidate_agent_runner import (
    LocalCandidateAgentRunner,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from tests.integration.evaluation.scripted_golden_orchestrator import (
    AuditRepository,
    ScriptedGoldenOrchestrator,
    WorkflowRepository,
)


@pytest.fixture
def eval_backend_safety_case() -> EvalCase:
    """Load the unchanged developer golden used by cancellation regressions."""
    suite = JsonlGoldenDatasetLoader().load(
        "backend_developer_development",
        Path("evals/datasets/backend_developer_development.jsonl"),
    )
    return next(case for case in suite.cases if case.id == "bd-dev-002")


@pytest.fixture
def golden_lifecycle_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[EvalCase, str], CandidateRunResult]:
    """Run public candidate observation with real offline workflow actions."""

    def run(case: EvalCase, implementation: str) -> CandidateRunResult:
        def orchestrator(
            database_path: Path,
            workflow_repository: WorkflowRepository,
            audit_repository: AuditRepository,
            case: EvalCase,
            settings: OllamaSettings,
        ) -> ScriptedGoldenOrchestrator:
            assert database_path == workflow_repository.database_path
            assert settings.model == "qwen3.5:9b"
            return ScriptedGoldenOrchestrator(
                workflow_repository,
                audit_repository,
                case,
                implementation,
            )

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )
        return asyncio.run(
            LocalCandidateAgentRunner(OllamaSettings()).run_case(
                case,
                "qwen3.5:9b",
                1,
            ),
        )

    return run


@pytest.fixture
def evaluation_case_record() -> dict[str, object]:
    """Provide a copy of an unchanged backend golden record."""
    path = Path("evals/datasets/backend_developer_development.jsonl")
    for line in path.read_text().splitlines():
        record: object = json.loads(line)
        assert isinstance(record, dict)
        values = cast("dict[str, object]", record)
        if values.get("id") == "bd-dev-002":
            return values
    raise AssertionError("Required backend golden case is missing.")


@pytest.fixture
def evaluation_result_record(tmp_path: Path) -> dict[str, object]:
    """Provide valid persisted evidence for corrupted-result tests."""
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    result = EvalRunResult(
        id="boundary-run",
        suite_id="backend_developer_development",
        candidate_model="qwen3.5:9b",
        judge_model=None,
        dataset_hash="dataset-hash",
        rubric_hash="rubric-hash",
        instructions_hash="instructions-hash",
        package_version="0.1.0",
        started_at=timestamp,
        ended_at=timestamp,
        case_results=(
            EvalCaseResult(
                case_id="bd-dev-002",
                repetition=1,
                candidate_result=CandidateRunResult(
                    role=DevelopmentRole.BACKEND_DEVELOPER,
                    model="qwen3.5:9b",
                    final_response="Task completed.",
                    tool_calls=(),
                    database_effects=(),
                ),
                deterministic_grade=DeterministicGrade(True, False, ()),
                judge_grade=None,
                verdict=EvalVerdict.PASS,
            ),
        ),
        warnings=(),
    )
    repository = JsonEvalResultRepository(tmp_path)
    repository.save(result)
    parsed: object = json.loads((tmp_path / "boundary-run.json").read_text())
    assert isinstance(parsed, dict)
    return cast("dict[str, object]", parsed)
