"""Capture durable candidate diagnostics before temporary state disappears."""

import asyncio
from dataclasses import asdict
from pathlib import Path

import pytest

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_candidate_execution_context import (
    EvalCandidateExecutionContext,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.agent_watchdog_settings import (
    AgentWatchdogSettings,
)
from agent_team.infrastructure.evaluation import local_candidate_agent_runner
from agent_team.infrastructure.evaluation.local_candidate_agent_runner import (
    LocalCandidateAgentRunner,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository,
)
from tests.integration.evaluation.cancel_after_audit_orchestrator import (
    CancelAfterAuditOrchestrator,
)

SQLiteAgentAuditRepository = (
    sqlite_agent_audit_repository.SQLiteAgentAuditRepository
)
SQLiteWorkflowRepository = sqlite_workflow_repository.SQLiteWorkflowRepository


class TestLocalCandidateCancellationCapture:
    """Keep uncertain outcomes distinct and propagate runtime watchdogs."""

    def test_candidate_watchdogs_reach_provider_and_task(
        self,
        eval_backend_safety_case: EvalCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass explicit safety values into provider composition."""
        tasks: list[AgentTask] = []
        watchdogs = AgentWatchdogSettings(
            provider_response_timeout_seconds=1500,
            segment_timeout_seconds=2400,
            no_advancement_timeout_seconds=3600,
        )

        def orchestrator(
            database_path: Path,
            workflow_repository: SQLiteWorkflowRepository,
            audit_repository: SQLiteAgentAuditRepository,
            case: EvalCase,
            settings: OllamaSettings,
        ) -> CancelAfterAuditOrchestrator:
            assert database_path == workflow_repository.database_path
            assert case.id == "bd-dev-002"
            assert settings.provider_response_timeout_seconds == 1500
            return CancelAfterAuditOrchestrator(
                workflow_repository,
                audit_repository,
                "completed",
                tasks,
            )

        monkeypatch.setattr(
            local_candidate_agent_runner, "_orchestrator", orchestrator
        )

        result = asyncio.run(
            LocalCandidateAgentRunner(OllamaSettings()).run_case(
                eval_backend_safety_case,
                "qwen3.5:9b",
                1,
                context=EvalCandidateExecutionContext(watchdogs=watchdogs),
            )
        )

        assert result.status == "completed"
        assert len(tasks) == 1
        assert tasks[0].watchdogs is watchdogs

    @pytest.mark.parametrize(
        "expectation",
        [
            ("before_patch", "in_progress", False, False),
            ("after_patch", "in_progress", True, False),
            ("before_handoff", "in_progress", True, False),
            ("after_handoff", "verification_pending", True, True),
        ],
    )
    def test_partial_audit_is_captured_before_cleanup(
        self,
        eval_backend_safety_case: EvalCase,
        monkeypatch: pytest.MonkeyPatch,
        expectation: tuple[str, str, bool, bool],
    ) -> None:
        """Checkpoint trusted patch and handoff outcomes without leaks."""
        boundary, status, has_patch, has_handoff = expectation
        tasks: list[AgentTask] = []
        snapshots: list[CandidateRunResult] = []
        databases: list[Path] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: SQLiteWorkflowRepository,
            audit_repository: SQLiteAgentAuditRepository,
            case: EvalCase,
            settings: OllamaSettings,
        ) -> CancelAfterAuditOrchestrator:
            assert case.id == "bd-dev-002"
            assert settings.model == "qwen3.5:9b"
            databases.append(database_path)
            return CancelAfterAuditOrchestrator(
                workflow_repository,
                audit_repository,
                boundary,
                tasks,
            )

        def checkpoint(candidate: CandidateRunResult) -> None:
            assert databases[0].exists()
            snapshots.append(candidate)

        monkeypatch.setattr(
            local_candidate_agent_runner, "_orchestrator", orchestrator
        )

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                LocalCandidateAgentRunner(OllamaSettings()).run_case(
                    eval_backend_safety_case,
                    "qwen3.5:9b",
                    1,
                    context=EvalCandidateExecutionContext(
                        checkpoint=checkpoint
                    ),
                )
            )

        assert len(tasks) == len(snapshots) == 1
        assert not databases[0].exists()
        snapshot = snapshots[0]
        assert snapshot.status == "interrupted"
        assert snapshot.termination_reason == "cancelled"
        assert snapshot.liveness_snapshot is not None
        assert snapshot.liveness_snapshot.segment_count == 2
        assert snapshot.liveness_snapshot.task_status == status
        assert bool(snapshot.liveness_snapshot.changed_paths) is has_patch
        assert (
            any(
                effect.table == "task_handoffs"
                for effect in snapshot.database_effects
            )
            is has_handoff
        )
        rendered = str(asdict(snapshot))
        assert "fixture-private-source" not in rendered
        assert "private-value" not in rendered
        assert "fixture-private-check-output" not in rendered
        assert "fixture-private-prompt" not in rendered
        assert "fixture-private-content" not in rendered
        assert str(databases[0].parent) not in rendered
        assert eval_backend_safety_case.user_input not in rendered

    @pytest.mark.parametrize(
        "boundary",
        ["provider_timeout", "segment_timeout", "cleanup_timeout"],
    )
    def test_provider_and_segment_timeouts_preserve_uncertain_effects(
        self,
        eval_backend_safety_case: EvalCase,
        monkeypatch: pytest.MonkeyPatch,
        boundary: str,
    ) -> None:
        """Capture distinct timeout classifications before audit cleanup."""
        snapshots: list[CandidateRunResult] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: SQLiteWorkflowRepository,
            audit_repository: SQLiteAgentAuditRepository,
            case: EvalCase,
            settings: OllamaSettings,
        ) -> CancelAfterAuditOrchestrator:
            _ = (database_path, case, settings)
            return CancelAfterAuditOrchestrator(
                workflow_repository, audit_repository, boundary, []
            )

        monkeypatch.setattr(
            local_candidate_agent_runner, "_orchestrator", orchestrator
        )
        result = asyncio.run(
            LocalCandidateAgentRunner(OllamaSettings()).run_case(
                eval_backend_safety_case,
                "local",
                1,
                context=EvalCandidateExecutionContext(
                    checkpoint=snapshots.append
                ),
            )
        )

        assert result.status == "timed_out"
        assert result.termination_reason == boundary
        assert snapshots == [result]
        assert result.liveness_snapshot is not None
        assert result.liveness_snapshot.changed_paths
        assert "fixture-private" not in str(asdict(result))

    def test_audit_liveness_fallback_excludes_unsafe_metadata(
        self,
        eval_backend_safety_case: EvalCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Require safe relative paths, known names and complete outcomes."""
        snapshots: list[CandidateRunResult] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: SQLiteWorkflowRepository,
            audit_repository: SQLiteAgentAuditRepository,
            case: EvalCase,
            settings: OllamaSettings,
        ) -> CancelAfterAuditOrchestrator:
            _ = (database_path, case, settings)
            return CancelAfterAuditOrchestrator(
                workflow_repository, audit_repository, "unsafe_metadata", []
            )

        monkeypatch.setattr(
            local_candidate_agent_runner, "_orchestrator", orchestrator
        )
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                LocalCandidateAgentRunner(OllamaSettings()).run_case(
                    eval_backend_safety_case,
                    "local",
                    1,
                    context=EvalCandidateExecutionContext(
                        checkpoint=snapshots.append
                    ),
                )
            )

        observed = snapshots[0].liveness_snapshot
        assert observed is not None
        assert observed.changed_paths == ("backend/auth_service.py",)
        assert observed.last_tool_name == "run_check"
        assert observed.last_check_outcome is None

    @pytest.mark.parametrize(
        "boundary",
        ["runtime_failure", "stalled", "infrastructure_failure"],
    )
    def test_failed_partial_checkpoints_omit_private_metadata(
        self,
        eval_backend_safety_case: EvalCase,
        monkeypatch: pytest.MonkeyPatch,
        boundary: str,
    ) -> None:
        """Sanitize unrecoverable failures as strictly as interruption."""
        snapshots: list[CandidateRunResult] = []
        databases: list[Path] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: SQLiteWorkflowRepository,
            audit_repository: SQLiteAgentAuditRepository,
            case: EvalCase,
            settings: OllamaSettings,
        ) -> CancelAfterAuditOrchestrator:
            _ = (case, settings)
            databases.append(database_path)
            return CancelAfterAuditOrchestrator(
                workflow_repository, audit_repository, boundary, []
            )

        def checkpoint(candidate: CandidateRunResult) -> None:
            assert databases[0].exists()
            snapshots.append(candidate)

        monkeypatch.setattr(
            local_candidate_agent_runner, "_orchestrator", orchestrator
        )
        result = asyncio.run(
            LocalCandidateAgentRunner(OllamaSettings()).run_case(
                eval_backend_safety_case,
                "local",
                1,
                context=EvalCandidateExecutionContext(checkpoint=checkpoint),
            )
        )

        assert result.status in {"failed", "infrastructure_error"}
        assert snapshots == [result]
        assert not databases[0].exists()
        rendered = str(asdict(result))
        assert "fixture-private-prompt" not in rendered
        assert "fixture-private-content" not in rendered
        assert "/private/workspace" not in rendered
        assert "private-error-key" not in rendered
        assert "private-title-key" not in rendered
        feature = next(
            effect
            for effect in result.database_effects
            if effect.table == "features"
        )
        assert len(str(feature.field_values["title"])) <= 160
