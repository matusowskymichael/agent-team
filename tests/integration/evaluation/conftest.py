"""Fixtures for offline developer golden lifecycle regressions."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.infrastructure.evaluation import local_candidate_agent_runner
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
