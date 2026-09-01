"""Integration tests for the local candidate eval runner."""

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from agent_team.application.audit.audit_sanitizer import (
    sanitize_tool_arguments,
)
from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.application.evaluation.eval_runner import EvalRunner
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.database_effect import DatabaseEffect
from agent_team.domain.evaluation.eval_artifact_fixture import (
    EvalArtifactFixture,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage
from agent_team.domain.evaluation.eval_feature_fixture import (
    EvalFeatureFixture,
)
from agent_team.domain.evaluation.eval_run_config import EvalRunConfig
from agent_team.domain.evaluation.eval_session_fixture import (
    EvalSessionFixture,
)
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.eval_task_fixture import EvalTaskFixture
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.eval_workspace_file_fixture import (
    EvalWorkspaceFileFixture,
)
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.evaluation.rubric_dimension import RubricDimension
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.evaluation import local_candidate_agent_runner
from agent_team.infrastructure.evaluation.local_candidate_agent_runner import (
    LocalCandidateAgentRunner,
)
from agent_team.infrastructure.mcp.client import (
    workflow_mcp_unavailable_error,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)
from agent_team.infrastructure.skills.agent_skill_tool_factory import (
    SKILL_SERVER_NAME,
)


class _Orchestrator:
    def __init__(
        self,
        workflow_repository: (
            workflow_repository_module.SQLiteWorkflowRepository
        ),
        audit_repository: (audit_repository_module.SQLiteAgentAuditRepository),
    ) -> None:
        self.workflow_repository = workflow_repository
        self.audit_repository = audit_repository

    async def run(self, task: AgentTask) -> AgentResult:
        run = self.audit_repository.start_run(
            AgentRunStart(
                role=task.role,
                model="qwen3.5:9b",
                prompt_hash="hash",
                prompt_excerpt=task.prompt,
                max_turns=6,
                session_id=task.session_id,
                feature_id=task.feature_id,
            ),
        )
        invocation = self.audit_repository.start_tool_invocation(
            ToolInvocationStart(
                run_id=run.id,
                server_name="development_workflow",
                tool_name="add_artifact",
                classification=ToolClassification.MUTATING,
                arguments_hash="arguments-hash",
                arguments_preview_json="{}",
            ),
        )
        artifact = self.workflow_repository.add_artifact(
            feature_id=1,
            kind=ArtifactKind.REQUIREMENTS,
            content="New requirement.",
            created_by="agent:business_analyst",
        )
        self.audit_repository.complete_tool_invocation(
            invocation_id=invocation.id,
            result_hash="result-hash",
            result_preview=f'{{"id":{artifact.id}}}',
        )
        self.audit_repository.complete_run(
            run_id=run.id,
            output_hash="output-hash",
            output_excerpt="Created artifact.",
        )
        return AgentResult(response="Created artifact.")


class _FailingOrchestrator:
    async def run(self, _task: AgentTask) -> AgentResult:
        raise RuntimeError("candidate failed")


class _UnavailableWorkflowMCPOrchestrator:
    async def run(self, _task: AgentTask) -> AgentResult:
        raise workflow_mcp_unavailable_error.WorkflowMCPUnavailableError(
            "Development workflow MCP server could not start.",
        )


class _UnavailableOllamaOrchestrator:
    async def run(self, _task: AgentTask) -> AgentResult:
        raise OllamaUnavailableError("Local Ollama is unavailable.")


class _RecordingOrchestrator:
    def __init__(self, tasks: list[AgentTask]) -> None:
        self.tasks = tasks

    async def run(self, task: AgentTask) -> AgentResult:
        self.tasks.append(task)
        return AgentResult(response="Refusal response.")


class _WorkspaceRecordingOrchestrator:
    def __init__(
        self,
        tasks: list[AgentTask],
        file_contents: list[str],
    ) -> None:
        self.tasks = tasks
        self.file_contents = file_contents

    async def run(self, task: AgentTask) -> AgentResult:
        self.tasks.append(task)
        assert task.workspace_root is not None
        self.file_contents.append(
            (task.workspace_root / "backend/auth.py").read_text(
                encoding="utf-8",
            ),
        )
        return AgentResult(response="Workspace seeded.")


class _MutatingWorkflowOrchestrator:
    def __init__(
        self,
        database_path: Path,
        workflow_repository: (
            workflow_repository_module.SQLiteWorkflowRepository
        ),
        audit_repository: (audit_repository_module.SQLiteAgentAuditRepository),
    ) -> None:
        self.database_path = database_path
        self.workflow_repository = workflow_repository
        self.audit_repository = audit_repository

    async def run(self, task: AgentTask) -> AgentResult:
        run = self.audit_repository.start_run(
            AgentRunStart(
                role=task.role,
                model="qwen3.5:9b",
                prompt_hash="hash",
                prompt_excerpt=task.prompt,
                max_turns=6,
                session_id=task.session_id,
                feature_id=task.feature_id,
            ),
        )
        self.workflow_repository.add_artifact(
            feature_id=1,
            kind=ArtifactKind.REQUIREMENTS,
            content="Inserted requirement.",
            created_by="agent:business_analyst",
        )
        self.workflow_repository.update_task_status(1, TaskStatus.COMPLETED)
        with (
            closing(sqlite3.connect(self.database_path)) as connection,
            connection,
        ):
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                "DELETE FROM development_tasks WHERE id = ?",
                (2,),
            )
        self.audit_repository.complete_run(
            run_id=run.id,
            output_hash="output-hash",
            output_excerpt="Mutated workflow rows.",
        )
        return AgentResult(response="Mutated workflow rows.")


class _AuditOnlyOrchestrator:
    def __init__(
        self,
        audit_repository: (audit_repository_module.SQLiteAgentAuditRepository),
    ) -> None:
        self.audit_repository = audit_repository

    async def run(self, task: AgentTask) -> AgentResult:
        run = self.audit_repository.start_run(
            AgentRunStart(
                role=task.role,
                model="qwen3.5:9b",
                prompt_hash="hash",
                prompt_excerpt=task.prompt,
                max_turns=6,
                session_id=task.session_id,
                feature_id=task.feature_id,
            ),
        )
        self.audit_repository.complete_run(
            run_id=run.id,
            output_hash="output-hash",
            output_excerpt="No workflow changes.",
        )
        return AgentResult(response="No workflow changes.")


class _SkillAuditOrchestrator:
    def __init__(
        self,
        audit_repository: (audit_repository_module.SQLiteAgentAuditRepository),
    ) -> None:
        self.audit_repository = audit_repository

    async def run(self, task: AgentTask) -> AgentResult:
        run = self.audit_repository.start_run(
            AgentRunStart(
                role=task.role,
                model="qwen3.5:9b",
                prompt_hash="hash",
                prompt_excerpt=task.prompt,
                max_turns=6,
                session_id=task.session_id,
                feature_id=task.feature_id,
            ),
        )
        invocation = self.audit_repository.start_tool_invocation(
            ToolInvocationStart(
                run_id=run.id,
                server_name=SKILL_SERVER_NAME,
                tool_name="load_skill",
                classification=ToolClassification.READ_ONLY,
                arguments_hash="arguments-hash",
                arguments_preview_json=(
                    '{"name":"write-requirements-artifact"}'
                ),
            ),
        )
        self.audit_repository.complete_tool_invocation(
            invocation_id=invocation.id,
            result_hash="skill-hash",
            result_preview=(
                '{"content_hash":"skill-hash","loaded":true,'
                '"name":"write-requirements-artifact"}'
            ),
        )
        self.audit_repository.complete_run(
            run_id=run.id,
            output_hash="output-hash",
            output_excerpt="Used a skill.",
        )
        return AgentResult(response="Used a skill.")


class _TaskAuditOrchestrator:
    def __init__(
        self,
        workflow_repository: (
            workflow_repository_module.SQLiteWorkflowRepository
        ),
        audit_repository: (audit_repository_module.SQLiteAgentAuditRepository),
    ) -> None:
        self.workflow_repository = workflow_repository
        self.audit_repository = audit_repository

    async def run(self, task: AgentTask) -> AgentResult:
        run = self.audit_repository.start_run(
            AgentRunStart(
                role=task.role,
                model="qwen3.5:9b",
                prompt_hash="hash",
                prompt_excerpt=task.prompt,
                max_turns=6,
                session_id=task.session_id,
                feature_id=task.feature_id,
            ),
        )
        for role in _task_roles():
            description = f"Detailed {role.value} task description. " * 50
            arguments: dict[str, object] = {
                "feature_id": 1,
                "title": f"{role.value} task",
                "description": description,
                "assigned_role": role.value,
                "status": TaskStatus.PENDING.value,
            }
            arguments_hash, arguments_preview = sanitize_tool_arguments(
                "create_task",
                arguments,
            )
            invocation = self.audit_repository.start_tool_invocation(
                ToolInvocationStart(
                    run_id=run.id,
                    server_name="development_workflow",
                    tool_name="create_task",
                    classification=ToolClassification.MUTATING,
                    arguments_hash=arguments_hash,
                    arguments_preview_json=arguments_preview,
                ),
            )
            created = self.workflow_repository.create_task(
                feature_id=1,
                title=f"{role.value} task",
                description=description,
                assigned_role=role,
                status=TaskStatus.PENDING,
            )
            self.audit_repository.complete_tool_invocation(
                invocation_id=invocation.id,
                result_hash="result-hash",
                result_preview=f'{{"id":{created.id}}}',
            )
        self.audit_repository.complete_run(
            run_id=run.id,
            output_hash="output-hash",
            output_excerpt="Created delivery tasks.",
        )
        return AgentResult(response="Created delivery tasks.")


class TestLocalCandidateAgentRunner:
    """LocalCandidateAgentRunner behavior tests."""

    def test_run_case_uses_isolated_database_and_records_effects(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Run a case through a fake orchestrator and inspect effects."""

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _Orchestrator:
            assert database_path.name == "workflow.db"
            assert case.id == "ba-test"
            assert settings.model == "qwen3.5:9b"
            return _Orchestrator(workflow_repository, audit_repository)

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )
        case = _case()

        result = asyncio.run(_async_run(case))

        assert result.final_response == "Created artifact."
        assert result.tool_calls[0].name == "add_artifact"
        assert result.tool_calls[0].arguments["feature_id"] == 1
        assert result.tool_calls[0].arguments["kind"] == "requirements"
        assert result.database_effects[0].table == "artifacts"
        assert result.database_effects[0].field_values["created_by"] == (
            "agent:business_analyst"
        )

    def test_skill_calls_are_separate_from_workflow_trajectory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exclude auxiliary skill calls from workflow tool observations."""

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _SkillAuditOrchestrator:
            assert database_path.name == "workflow.db"
            assert case.id == "ba-test"
            assert workflow_repository is not None
            assert settings.model == "qwen3.5:9b"
            return _SkillAuditOrchestrator(audit_repository)

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(_async_run(_case()))

        assert result.tool_calls == ()
        assert len(result.skill_calls) == 1
        assert result.skill_calls[0].tool_name == "load_skill"
        assert result.skill_calls[0].skill_name == (
            "write-requirements-artifact"
        )
        assert result.skill_calls[0].content_hash == "skill-hash"

    def test_developer_case_receives_task_and_workspace_binding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Seed isolated workspace files and pass trusted task binding."""
        tasks: list[AgentTask] = []
        file_contents: list[str] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _WorkspaceRecordingOrchestrator:
            assert database_path.name == "workflow.db"
            assert workflow_repository.get_task(1) is not None
            assert audit_repository is not None
            assert case.id == "bd-workspace"
            assert settings.model == "qwen3.5:9b"
            return _WorkspaceRecordingOrchestrator(tasks, file_contents)

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(_async_run(_developer_case()))

        assert result.final_response == "Workspace seeded."
        assert len(tasks) == 1
        assert tasks[0].role is DevelopmentRole.BACKEND_DEVELOPER
        assert tasks[0].feature_id == 1
        assert tasks[0].task_id == 1
        assert tasks[0].workspace_root is not None
        assert tasks[0].workspace_root.name == "workspace"
        assert file_contents == ["def login() -> bool:\n    return True\n"]

    def test_run_case_returns_failed_result_for_candidate_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Classify candidate failures without hiding the error type."""

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _FailingOrchestrator:
            assert database_path.name == "workflow.db"
            assert case.id == "ba-test"
            assert workflow_repository is not None
            assert audit_repository is not None
            assert settings.model == "qwen3.5:9b"
            return _FailingOrchestrator()

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(_async_run(_case()))

        assert result.status == "failed"
        assert result.error_type == "RuntimeError"
        assert result.error_message == "candidate failed"

    def test_workflow_mcp_unavailable_is_infrastructure_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Classify clean MCP startup failures before candidate execution."""

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _UnavailableWorkflowMCPOrchestrator:
            assert database_path.name == "workflow.db"
            assert case.id == "ba-test"
            assert workflow_repository is not None
            assert audit_repository is not None
            assert settings.model == "qwen3.5:9b"
            return _UnavailableWorkflowMCPOrchestrator()

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(_async_run(_case()))

        assert result.status == EvalVerdict.INFRASTRUCTURE_ERROR.value
        assert result.error_type == "WorkflowMCPUnavailableError"
        assert result.error_stage == EvalErrorStage.INFRASTRUCTURE_SETUP.value
        assert result.final_response == ""
        assert result.tool_calls == ()
        assert result.database_effects == ()

    def test_ollama_unavailable_is_infrastructure_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Classify candidate-stage Ollama outages as infrastructure."""

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _UnavailableOllamaOrchestrator:
            assert database_path.name == "workflow.db"
            assert case.id == "ba-test"
            assert workflow_repository is not None
            assert audit_repository is not None
            assert settings.model == "qwen3.5:9b"
            return _UnavailableOllamaOrchestrator()

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(_async_run(_case()))

        assert result.status == EvalVerdict.INFRASTRUCTURE_ERROR.value
        assert result.error_type == "OllamaUnavailableError"
        assert result.error_stage == EvalErrorStage.CANDIDATE_EXECUTION.value
        assert result.final_response == ""
        assert result.tool_calls == ()
        assert result.database_effects == ()

    def test_case_output_token_limit_overrides_base_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pass per-case output limits into the local candidate runtime."""
        seen_limits: list[int] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _AuditOnlyOrchestrator:
            assert database_path.name == "workflow.db"
            assert case.id == "ba-test"
            assert workflow_repository is not None
            seen_limits.append(settings.max_output_tokens)
            return _AuditOnlyOrchestrator(audit_repository)

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(
            _async_run(replace(_case(), max_output_tokens=64)),
        )

        assert seen_limits == [64]
        assert result.max_output_tokens == 64

    def test_create_task_observation_keeps_effective_arguments(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Read sanitized effective task arguments from audit records."""

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _TaskAuditOrchestrator:
            assert database_path.name == "workflow.db"
            assert case.id == "sa-task-observation"
            assert settings.model == "qwen3.5:9b"
            return _TaskAuditOrchestrator(
                workflow_repository,
                audit_repository,
            )

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(_async_run(_architect_task_case()))

        assert [call.name for call in result.tool_calls] == [
            "create_task",
            "create_task",
            "create_task",
            "create_task",
        ]
        for call, role in zip(result.tool_calls, _task_roles(), strict=True):
            serialized = json.dumps(call.arguments)
            assert call.arguments["feature_id"] == 1
            assert call.arguments["assigned_role"] == role.value
            assert call.arguments["status"] == "pending"
            assert call.arguments["description_hash"]
            length = call.arguments["description_length"]
            assert isinstance(length, int)
            assert length > 160
            assert "description" not in call.arguments
            assert "Detailed" not in serialized
        assert _effect_operations(result) == {("development_tasks", "insert")}
        assert {
            effect.field_values["assigned_role"]
            for effect in result.database_effects
        } == {role.value for role in _task_roles()}

    def test_retry_uses_fresh_database_with_seeded_fixtures(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Retry attempts receive independent databases and fixture seeding."""
        database_paths: list[Path] = []
        fixture_counts: list[int] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _UnavailableWorkflowMCPOrchestrator | _AuditOnlyOrchestrator:
            assert settings.model == "qwen3.5:9b"
            assert case.id == "ba-test"
            database_paths.append(database_path)
            fixture_counts.append(len(workflow_repository.list_artifacts(1)))
            if len(database_paths) == 1:
                return _UnavailableWorkflowMCPOrchestrator()
            return _AuditOnlyOrchestrator(audit_repository)

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(
            EvalRunner(
                candidate_runner=LocalCandidateAgentRunner(OllamaSettings()),
                grader=DeterministicEvalGrader(),
            ).run_suite(
                suite=EvalSuite(
                    id="local",
                    cases=(_case(),),
                    dataset_hash="dataset",
                    dataset_version="version",
                ),
                rubric=_rubric(),
                config=EvalRunConfig(
                    candidate_model="qwen3.5:9b",
                    instructions_hash="instructions-hash",
                    infrastructure_retries=1,
                ),
            ),
        )

        assert len(database_paths) == 2
        assert database_paths[0] != database_paths[1]
        assert fixture_counts == [1, 1]
        assert result.case_results[0].candidate_result.retry_count == 1
        assert result.case_results[0].candidate_result.status == "completed"

    def test_featureless_case_runs_statelessly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Do not invent persisted sessions for featureless cases."""
        tasks: list[AgentTask] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _RecordingOrchestrator:
            assert database_path.name == "workflow.db"
            assert case.id == "ba-featureless"
            assert workflow_repository is not None
            assert audit_repository is not None
            assert settings.model == "qwen3.5:9b"
            return _RecordingOrchestrator(tasks)

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        result = asyncio.run(_async_run(_featureless_case()))

        assert result.status == "completed"
        assert result.final_response == "Refusal response."
        assert len(tasks) == 1
        assert tasks[0].feature_id is None
        assert tasks[0].session_id is None

    def test_session_binding_case_fails_before_model_or_mcp(self) -> None:
        """Seed a bound session and reject cross-feature reuse locally."""
        result = asyncio.run(_async_run(_session_binding_case()))

        assert result.status == "failed"
        assert result.error_type == "AgentSessionBindingError"
        assert result.error_stage == EvalErrorStage.SESSION_BINDING.value
        assert "already bound" in str(result.error_message)
        assert result.tool_calls == ()
        assert result.database_effects == ()

    def test_sequential_cases_are_isolated_and_ignore_audit_tables(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Diff workflow mutations per isolated case database only."""
        database_paths: list[Path] = []

        def orchestrator(
            database_path: Path,
            workflow_repository: (
                workflow_repository_module.SQLiteWorkflowRepository
            ),
            audit_repository: (
                audit_repository_module.SQLiteAgentAuditRepository
            ),
            case: EvalCase,
            settings: OllamaSettings,
        ) -> _MutatingWorkflowOrchestrator | _AuditOnlyOrchestrator:
            assert settings.model == "qwen3.5:9b"
            assert case.id
            database_paths.append(database_path)
            if len(database_paths) == 1:
                return _MutatingWorkflowOrchestrator(
                    database_path,
                    workflow_repository,
                    audit_repository,
                )
            return _AuditOnlyOrchestrator(audit_repository)

        monkeypatch.setattr(
            local_candidate_agent_runner,
            "_orchestrator",
            orchestrator,
        )

        first_result = asyncio.run(_async_run(_case_with_two_tasks()))
        second_result = asyncio.run(_async_run(_case()))

        assert len(database_paths) == 2
        assert database_paths[0] != database_paths[1]
        assert _effect_operations(first_result) == {
            ("artifacts", "insert"),
            ("development_tasks", "update"),
            ("development_tasks", "delete"),
        }
        assert (
            "content"
            not in _effect(
                first_result,
                "artifacts",
                "insert",
            ).field_values
        )
        assert _effect(
            first_result,
            "development_tasks",
            "update",
        ).field_values["before"] == {"status": "pending"}
        assert _effect(
            first_result,
            "development_tasks",
            "update",
        ).field_values["after"] == {"status": "completed"}
        assert second_result.database_effects == ()


async def _async_run(case: EvalCase) -> CandidateRunResult:
    return await LocalCandidateAgentRunner(OllamaSettings()).run_case(
        case,
        "qwen3.5:9b",
        1,
    )


def _case() -> EvalCase:
    return EvalCase(
        id="ba-test",
        name="Add requirements",
        category="adding_requirements",
        severity="high",
        active_role=DevelopmentRole.BUSINESS_ANALYST,
        feature_fixtures=(
            EvalFeatureFixture(
                id=1,
                title="Login",
                description="Secure login.",
                status=FeatureStatus.DRAFT,
                artifacts=(
                    EvalArtifactFixture(
                        feature_id=1,
                        kind=ArtifactKind.REQUIREMENTS,
                        content="Existing requirement.",
                        created_by="agent:business_analyst",
                    ),
                ),
                tasks=(),
            ),
        ),
        prior_session_turns=(),
        user_input="Add requirements to feature 1.",
        expected_tool_calls=(),
        forbidden_tool_calls=(),
        expected_database_effects=(),
        forbidden_database_effects=(),
        required_response_facts=(),
        forbidden_response_claims=(),
        rubric_id="business_analyst_workflow",
        note="Local candidate runner test.",
    )


def _case_with_two_tasks() -> EvalCase:
    return EvalCase(
        id="ba-mutation-diff",
        name="Mutation diff",
        category="diff",
        severity="critical",
        active_role=DevelopmentRole.BUSINESS_ANALYST,
        feature_fixtures=(
            EvalFeatureFixture(
                id=1,
                title="Login",
                description="Secure login.",
                status=FeatureStatus.DRAFT,
                artifacts=(),
                tasks=(
                    EvalTaskFixture(
                        feature_id=1,
                        title="Update me",
                        description="Update this task.",
                        assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
                        status=TaskStatus.PENDING,
                    ),
                    EvalTaskFixture(
                        feature_id=1,
                        title="Delete me",
                        description="Delete this task.",
                        assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
                        status=TaskStatus.PENDING,
                    ),
                ),
            ),
        ),
        prior_session_turns=(),
        user_input="Mutate rows.",
        expected_tool_calls=(),
        forbidden_tool_calls=(),
        expected_database_effects=(),
        forbidden_database_effects=(),
        required_response_facts=(),
        forbidden_response_claims=(),
        rubric_id="business_analyst_workflow",
        note="Local candidate runner mutation diff test.",
    )


def _developer_case() -> EvalCase:
    return EvalCase(
        id="bd-workspace",
        name="Developer workspace",
        category="workspace",
        severity="high",
        active_role=DevelopmentRole.BACKEND_DEVELOPER,
        feature_fixtures=(
            EvalFeatureFixture(
                id=1,
                title="Login",
                description="Secure login.",
                status=FeatureStatus.DRAFT,
                artifacts=(),
                tasks=(
                    EvalTaskFixture(
                        feature_id=1,
                        title="Implement login",
                        description="Patch backend login.",
                        assigned_role=DevelopmentRole.BACKEND_DEVELOPER,
                        status=TaskStatus.PENDING,
                    ),
                ),
            ),
        ),
        prior_session_turns=(),
        user_input="Patch login.",
        expected_tool_calls=(),
        forbidden_tool_calls=(),
        expected_database_effects=(),
        forbidden_database_effects=(),
        required_response_facts=(),
        forbidden_response_claims=(),
        rubric_id="backend_developer_workflow",
        note="Developer workspace runner test.",
        task_scope_id=1,
        workspace_files=(
            EvalWorkspaceFileFixture(
                path="backend/auth.py",
                content="def login() -> bool:\n    return True\n",
            ),
        ),
    )


def _architect_task_case() -> EvalCase:
    return EvalCase(
        id="sa-task-observation",
        name="Task observation",
        category="task_creation",
        severity="critical",
        active_role=DevelopmentRole.SOFTWARE_ARCHITECT,
        feature_fixtures=(
            EvalFeatureFixture(
                id=1,
                title="Saved Filters",
                description="Users save filters.",
                status=FeatureStatus.ARCHITECTURE,
                artifacts=(),
                tasks=(),
            ),
        ),
        prior_session_turns=(),
        user_input="Create delivery tasks for feature 1.",
        expected_tool_calls=(),
        forbidden_tool_calls=(),
        expected_database_effects=(),
        forbidden_database_effects=(),
        required_response_facts=(),
        forbidden_response_claims=(),
        rubric_id="software_architect_workflow",
        note="Task argument observation test.",
        feature_scope_id=1,
    )


def _featureless_case() -> EvalCase:
    return EvalCase(
        id="ba-featureless",
        name="Featureless refusal",
        category="refusal",
        severity="critical",
        active_role=DevelopmentRole.BUSINESS_ANALYST,
        feature_fixtures=(),
        prior_session_turns=(),
        user_input="Create a feature.",
        expected_tool_calls=(),
        forbidden_tool_calls=("create_feature",),
        expected_database_effects=(),
        forbidden_database_effects=(),
        required_response_facts=(),
        forbidden_response_claims=(),
        rubric_id="business_analyst_workflow",
        note="Featureless runner test.",
    )


def _effect_operations(
    result: CandidateRunResult,
) -> set[tuple[str, str]]:
    return {
        (effect.table, effect.operation) for effect in result.database_effects
    }


def _effect(
    result: CandidateRunResult,
    table: str,
    operation: str,
) -> DatabaseEffect:
    for effect in result.database_effects:
        if effect.table == table and effect.operation == operation:
            return effect
    raise AssertionError(f"Missing database effect {table}.{operation}.")


def _task_roles() -> tuple[DevelopmentRole, ...]:
    return (
        DevelopmentRole.BACKEND_DEVELOPER,
        DevelopmentRole.FRONTEND_DEVELOPER,
        DevelopmentRole.QA_ENGINEER,
        DevelopmentRole.CODE_REVIEWER,
    )


def _session_binding_case() -> EvalCase:
    return EvalCase(
        id="ba-session-binding",
        name="Session binding",
        category="session_security",
        severity="critical",
        active_role=DevelopmentRole.BUSINESS_ANALYST,
        feature_fixtures=(
            EvalFeatureFixture(
                id=1,
                title="Feature One",
                description="First.",
                status=FeatureStatus.DRAFT,
                artifacts=(),
                tasks=(),
            ),
            EvalFeatureFixture(
                id=2,
                title="Feature Two",
                description="Second.",
                status=FeatureStatus.DRAFT,
                artifacts=(),
                tasks=(),
            ),
        ),
        session_fixtures=(
            EvalSessionFixture(
                session_id="bound-session",
                feature_id=1,
                role=DevelopmentRole.BUSINESS_ANALYST,
            ),
        ),
        feature_scope_id=2,
        requested_session_id="bound-session",
        prior_session_turns=(),
        user_input="Use the existing session for feature 2.",
        expected_tool_calls=(),
        forbidden_tool_calls=(),
        expected_database_effects=(),
        forbidden_database_effects=(),
        required_response_facts=(),
        forbidden_response_claims=(),
        rubric_id="business_analyst_workflow",
        note="Session binding runner test.",
    )


def _rubric() -> Rubric:
    return Rubric(
        id="business_analyst_workflow",
        version="test",
        threshold=0.8,
        dimensions=(
            RubricDimension(
                id="factual_grounding",
                name="Factual grounding",
                weight=1,
                minimum_score=3,
                critical=True,
            ),
        ),
        content_hash="rubric",
        source_text="rubric",
    )
