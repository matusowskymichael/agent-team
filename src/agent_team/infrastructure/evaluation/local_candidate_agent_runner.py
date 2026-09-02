"""Local candidate agent runner for evaluations."""

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from agent_team.application.audit.audit_sanitizer import sanitize_error
from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.application.runtime.orchestrator import Orchestrator
from agent_team.application.sessions.agent_session_service import (
    AgentSessionService,
)
from agent_team.application.skills.agent_skill_authorizer import (
    AgentSkillAuthorizer,
)
from agent_team.application.skills.agent_skill_service import (
    AgentSkillService,
)
from agent_team.application.workspace.workspace_service import (
    WorkspaceService,
)
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.database_effect import DatabaseEffect
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage
from agent_team.domain.evaluation.eval_feature_fixture import (
    EvalFeatureFixture,
)
from agent_team.domain.evaluation.eval_session_fixture import (
    EvalSessionFixture,
)
from agent_team.domain.evaluation.eval_workspace_file_fixture import (
    EvalWorkspaceFileFixture,
)
from agent_team.domain.evaluation.observed_skill_call import (
    ObservedSkillCall,
)
from agent_team.domain.evaluation.observed_tool_call import ObservedToolCall
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.sessions.agent_session_binding_error import (
    AgentSessionBindingError,
)
from agent_team.infrastructure.configuration.workflow_database_path import (
    AGENT_TEAM_DB_PATH_ENV,
)
from agent_team.infrastructure.evaluation.eval_hashes import hash_text_value
from agent_team.infrastructure.evaluation.evaluation_context_provider import (
    EvaluationContextProvider,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_process_options as mcp_process_options,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_config as workflow_mcp_config,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_factory as workflow_mcp_factory,
)
from agent_team.infrastructure.mcp.client import (
    workflow_mcp_unavailable_error,
)
from agent_team.infrastructure.ollama.ollama_agent_executor import (
    OllamaAgentExecutor,
)
from agent_team.infrastructure.ollama.ollama_model_factory import (
    create_ollama_model,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)
from agent_team.infrastructure.skills.agent_skill_tool_factory import (
    SKILL_SERVER_NAME,
    AgentSkillToolFactory,
)
from agent_team.infrastructure.skills.filesystem_agent_skill_catalog import (
    FilesystemAgentSkillCatalog,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)
from agent_team.infrastructure.workspace.workspace_tool_factory import (
    WorkspaceToolFactory,
)

from ..persistence.sqlite.audit.sqlite_agent_audit_repository import (
    SQLiteAgentAuditRepository,
)
from ..persistence.sqlite.sessions.sqlite_agent_session_repository import (
    SQLiteAgentSessionRepository,
)
from ..persistence.sqlite.sessions.sqlite_session_factory import (
    SQLiteSessionFactory,
)
from ..persistence.sqlite.workflow.sqlite_workflow_repository import (
    SQLiteWorkflowRepository,
)

WorkflowRow = dict[str, object]
WorkflowSnapshot = dict[str, dict[int, WorkflowRow]]
WORKFLOW_TABLES = ("features", "artifacts", "development_tasks")


@dataclass(frozen=True, slots=True)
class LocalCandidateAgentRunner:
    """Run candidate agents against isolated local workflow databases."""

    base_settings: OllamaSettings

    async def run_case(
        self,
        case: EvalCase,
        candidate_model: str,
        repetition: int,
    ) -> CandidateRunResult:
        """Run one evaluation case against local Ollama and MCP."""
        with TemporaryDirectory() as directory:
            temp_root = Path(directory)
            database_path = temp_root / "workflow.db"
            workspace_root = temp_root / "workspace"
            workflow_repository = SQLiteWorkflowRepository(database_path)
            _seed_fixtures(workflow_repository, case.feature_fixtures)
            _seed_sessions(database_path, case.session_fixtures)
            _seed_workspace(workspace_root, case.workspace_files)
            baseline = _snapshot(workflow_repository)
            audit_repository = SQLiteAgentAuditRepository(database_path)
            effective_settings = OllamaSettings(
                base_url=self.base_settings.base_url,
                model=candidate_model,
                max_output_tokens=(
                    case.max_output_tokens
                    or self.base_settings.max_output_tokens
                ),
                thinking_enabled=self.base_settings.thinking_enabled,
            )
            orchestrator = _orchestrator(
                database_path=database_path,
                workflow_repository=workflow_repository,
                audit_repository=audit_repository,
                case=case,
                settings=effective_settings,
            )
            try:
                result = await orchestrator.run(
                    AgentTask(
                        prompt=case.user_input,
                        role=case.active_role,
                        feature_id=_feature_scope(case),
                        session_id=_session_id(case, repetition),
                        task_id=case.task_scope_id,
                        workspace_root=_workspace_root(case, workspace_root),
                    ),
                )
                response = result.response
                status = "completed"
                error_type = None
                error_message = None
            except Exception as error:
                response = ""
                error_type, error_message = sanitize_error(error)
                status = _status(error)
                error_stage = _error_stage(error)
            else:
                error_stage = None

            effects = _database_effects(workflow_repository, baseline)
            tool_calls = _tool_calls(audit_repository, effects, case)
            skill_calls = _skill_calls(audit_repository)
            return CandidateRunResult(
                role=case.active_role,
                model=candidate_model,
                final_response=response,
                tool_calls=tool_calls,
                database_effects=effects,
                skill_calls=skill_calls,
                status=status,
                error_type=error_type,
                error_message=error_message,
                error_stage=error_stage,
                max_output_tokens=effective_settings.max_output_tokens,
            )


def _orchestrator(
    database_path: Path,
    workflow_repository: SQLiteWorkflowRepository,
    audit_repository: SQLiteAgentAuditRepository,
    case: EvalCase,
    settings: OllamaSettings,
) -> Orchestrator:
    authorizer = CapabilityAuthorizer(repository=workflow_repository)
    skill_service = AgentSkillService(
        catalog=FilesystemAgentSkillCatalog(),
        authorizer=AgentSkillAuthorizer(),
    )
    skill_tool_factory = AgentSkillToolFactory(
        service=skill_service,
        audit_repository=audit_repository,
    )
    workspace_tool_factory = WorkspaceToolFactory(
        service_factory=lambda workspace_root: WorkspaceService(
            repository=workflow_repository,
            executor=LocalWorkspaceExecutor(
                root=workspace_root,
                check_commands=_evaluation_check_commands(),
            ),
        ),
        audit_repository=audit_repository,
    )
    session_repository = SQLiteAgentSessionRepository(database_path)
    session_factory = SQLiteSessionFactory(database_path)
    environment = dict(os.environ)
    environment[AGENT_TEAM_DB_PATH_ENV] = str(database_path)
    process_options = mcp_process_options.DevelopmentWorkflowMCPProcessOptions(
        environ=environment,
    )
    runtime = OllamaAgentExecutor(
        model=create_ollama_model(settings),
        settings=settings,
        mcp_server_factory=lambda profile, run, task: (
            workflow_mcp_factory.create_development_workflow_mcp_server(
                workflow_mcp_config.DevelopmentWorkflowMCPServerConfig(
                    profile=profile,
                    authorizer=authorizer,
                    audit_repository=audit_repository,
                    run=run,
                    bound_task_id=task.task_id,
                ),
                process_options=process_options,
            ),
        ),
        skill_tool_factory=skill_tool_factory.create_tools,
        workspace_tool_factory=workspace_tool_factory.create_tools,
        session_factory=session_factory.create_session,
    )
    return Orchestrator(
        agent_executor=AgentHarness(
            runtime=runtime,
            audit_repository=audit_repository,
            session_service=AgentSessionService(session_repository),
            context_provider=EvaluationContextProvider(
                repository=workflow_repository,
                context_policy=case.context_policy,
            ),
            skill_service=skill_service,
        ),
    )


def _seed_fixtures(
    repository: SQLiteWorkflowRepository,
    features: tuple[EvalFeatureFixture, ...],
) -> None:
    for fixture in sorted(features, key=lambda feature: feature.id):
        feature = repository.create_feature(
            title=fixture.title,
            description=fixture.description,
            status=fixture.status,
        )
        if feature.id != fixture.id:
            raise ValueError("Feature fixture IDs must start at 1 in order.")
        for artifact in fixture.artifacts:
            repository.add_artifact(
                feature_id=artifact.feature_id,
                kind=artifact.kind,
                content=artifact.content,
                created_by=artifact.created_by,
            )
        for task in fixture.tasks:
            repository.create_task(
                feature_id=task.feature_id,
                title=task.title,
                description=task.description,
                assigned_role=task.assigned_role,
                status=task.status,
            )


def _seed_sessions(
    database_path: Path,
    sessions: tuple[EvalSessionFixture, ...],
) -> None:
    repository = SQLiteAgentSessionRepository(database_path)
    for session in sessions:
        repository.create_session(
            session_id=session.session_id,
            feature_id=session.feature_id,
            role=session.role,
        )


def _seed_workspace(
    workspace_root: Path,
    files: tuple[EvalWorkspaceFileFixture, ...],
) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    for file_fixture in files:
        path = Path(file_fixture.path.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Workspace fixture path must stay relative.")
        target = workspace_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_fixture.content, encoding="utf-8")


def _feature_scope(case: EvalCase) -> int | None:
    if case.feature_scope_id is not None:
        return case.feature_scope_id
    if not case.feature_fixtures:
        return None
    return case.feature_fixtures[0].id


def _session_id(case: EvalCase, repetition: int) -> str | None:
    if case.requested_session_id is not None:
        return case.requested_session_id
    if _feature_scope(case) is None:
        return None
    return f"eval-{case.id}-{repetition}"


def _workspace_root(case: EvalCase, workspace_root: Path) -> Path | None:
    if case.workspace_files or case.task_scope_id is not None:
        return workspace_root
    if case.active_role in {
        DevelopmentRole.BACKEND_DEVELOPER,
        DevelopmentRole.FRONTEND_DEVELOPER,
    }:
        return workspace_root
    return None


def _evaluation_check_commands() -> dict[str, tuple[str, ...]]:
    return {
        "backend": (sys.executable, "-c", "pass"),
        "frontend": (sys.executable, "-c", "pass"),
        "pytest": (sys.executable, "-c", "pass"),
        "pyright": (sys.executable, "-c", "pass"),
        "ruff": (sys.executable, "-c", "pass"),
    }


def _error_stage(error: Exception) -> str:
    if isinstance(error, AgentSessionBindingError):
        return EvalErrorStage.SESSION_BINDING.value
    if isinstance(
        error,
        workflow_mcp_unavailable_error.WorkflowMCPUnavailableError,
    ):
        return EvalErrorStage.INFRASTRUCTURE_SETUP.value
    return EvalErrorStage.CANDIDATE_EXECUTION.value


def _status(error: Exception) -> str:
    if isinstance(
        error,
        (
            workflow_mcp_unavailable_error.WorkflowMCPUnavailableError,
            OllamaUnavailableError,
        ),
    ):
        return "infrastructure_error"
    return "failed"


def _snapshot(
    repository: SQLiteWorkflowRepository,
) -> WorkflowSnapshot:
    features: dict[int, WorkflowRow] = {
        feature.id: {
            "id": feature.id,
            "title": feature.title,
            "description_hash": hash_text_value(feature.description),
            "description_length": len(feature.description),
            "status": feature.status.value,
        }
        for feature in repository.list_features()
    }
    artifacts: dict[int, WorkflowRow] = {
        artifact.id: {
            "id": artifact.id,
            "feature_id": artifact.feature_id,
            "kind": artifact.kind.value,
            "content_hash": hash_text_value(artifact.content),
            "content_length": len(artifact.content),
            "created_by": artifact.created_by,
        }
        for feature_id in features
        for artifact in repository.list_artifacts(feature_id)
    }
    tasks: dict[int, WorkflowRow] = {
        task.id: {
            "id": task.id,
            "feature_id": task.feature_id,
            "title": task.title,
            "description_hash": hash_text_value(task.description),
            "description_length": len(task.description),
            "assigned_role": task.assigned_role.value,
            "status": task.status.value,
        }
        for feature_id in features
        for task in repository.list_tasks(feature_id)
    }
    return {
        "features": features,
        "artifacts": artifacts,
        "development_tasks": tasks,
    }


def _tool_calls(
    audit_repository: SQLiteAgentAuditRepository,
    effects: tuple[DatabaseEffect, ...],
    case: EvalCase,
) -> tuple[ObservedToolCall, ...]:
    runs = audit_repository.list_runs(limit=1)
    if not runs:
        return ()
    invocations = audit_repository.list_tool_invocations(runs[0].id)
    artifact_effects = [
        effect
        for effect in effects
        if effect.table == "artifacts" and effect.operation == "insert"
    ]
    artifact_index = 0
    tool_calls: list[ObservedToolCall] = []
    for invocation in invocations:
        if invocation.server_name == SKILL_SERVER_NAME:
            continue
        arguments = _json_object(invocation.arguments_preview_json)
        arguments = _restore_expected_symbol_name(
            case,
            invocation.tool_name,
            arguments,
        )
        if invocation.tool_name == "add_artifact":
            arguments, artifact_index = _artifact_arguments(
                arguments,
                artifact_effects,
                artifact_index,
            )
        tool_calls.append(
            ObservedToolCall(
                name=invocation.tool_name,
                arguments=arguments,
                status=invocation.status.value,
                reached_mcp=invocation.status.value != "denied",
            )
        )
    return tuple(tool_calls)


def _restore_expected_symbol_name(
    case: EvalCase,
    tool_name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    """Restore a golden symbol name only when its audit hash matches."""
    if tool_name != "find_symbol":
        return arguments
    observed_hash = arguments.get("name_hash")
    if not isinstance(observed_hash, str):
        return arguments
    for expected_name in _expected_symbol_names(case):
        if hash_text_value(expected_name) == observed_hash:
            return {**arguments, "name": expected_name}
    return arguments


def _expected_symbol_names(case: EvalCase) -> tuple[str, ...]:
    expected_calls = list(case.expected_tool_calls)
    for trajectory in case.acceptable_tool_trajectories:
        expected_calls.extend(trajectory.required_tool_calls)
    return tuple(
        name
        for call in expected_calls
        if call.name == "find_symbol"
        and isinstance(name := call.arguments_subset.get("name"), str)
    )


def _skill_calls(
    audit_repository: SQLiteAgentAuditRepository,
) -> tuple[ObservedSkillCall, ...]:
    runs = audit_repository.list_runs(limit=1)
    if not runs:
        return ()
    invocations = audit_repository.list_tool_invocations(runs[0].id)
    skill_calls: list[ObservedSkillCall] = []
    for invocation in invocations:
        if invocation.server_name != SKILL_SERVER_NAME:
            continue
        arguments = _json_object(invocation.arguments_preview_json)
        result = _json_object(invocation.result_preview or "{}")
        skill_calls.append(
            ObservedSkillCall(
                tool_name=invocation.tool_name,
                skill_name=_skill_name(arguments, result),
                status=invocation.status.value,
                content_hash=_optional_text(result.get("content_hash")),
                resource_name=_optional_text(result.get("resource_name")),
            ),
        )
    return tuple(skill_calls)


def _skill_name(
    arguments: dict[str, object],
    result: dict[str, object],
) -> str:
    for key in ("name", "skill_name"):
        value = result.get(key, arguments.get(key))
        if isinstance(value, str):
            return value
    return ""


def _artifact_arguments(
    arguments: dict[str, object],
    effects: list[DatabaseEffect],
    artifact_index: int,
) -> tuple[dict[str, object], int]:
    if {"feature_id", "kind"}.issubset(arguments):
        return arguments, artifact_index
    if artifact_index >= len(effects):
        return arguments, artifact_index
    effect = effects[artifact_index]
    recovered = dict(arguments)
    for key in ("feature_id", "kind", "created_by"):
        if key in effect.field_values:
            recovered[key] = effect.field_values[key]
    return recovered, artifact_index + 1


def _database_effects(
    repository: SQLiteWorkflowRepository,
    baseline: WorkflowSnapshot,
) -> tuple[DatabaseEffect, ...]:
    current = _snapshot(repository)
    effects: list[DatabaseEffect] = []
    for table_name in WORKFLOW_TABLES:
        before_rows = baseline[table_name]
        after_rows = current[table_name]

        for row_id in sorted(after_rows):
            if row_id not in before_rows:
                effects.append(
                    DatabaseEffect(
                        table=table_name,
                        operation="insert",
                        field_values=dict(after_rows[row_id]),
                    ),
                )

        for row_id in sorted(before_rows):
            if row_id not in after_rows:
                effects.append(
                    DatabaseEffect(
                        table=table_name,
                        operation="delete",
                        field_values=dict(before_rows[row_id]),
                    ),
                )

        for row_id in sorted(before_rows.keys() & after_rows.keys()):
            before = before_rows[row_id]
            after = after_rows[row_id]
            if before != after:
                effects.append(
                    DatabaseEffect(
                        table=table_name,
                        operation="update",
                        field_values=_updated_field_values(before, after),
                    ),
                )

    return tuple(effects)


def _updated_field_values(
    before: WorkflowRow,
    after: WorkflowRow,
) -> dict[str, object]:
    before_changes = _changed_values(before, after)
    after_changes = _changed_values(after, before)
    field_values: dict[str, object] = {
        "id": after["id"],
        "before": before_changes,
        "after": after_changes,
    }
    for key in _ROW_IDENTITY_FIELDS:
        if key in after:
            field_values[key] = after[key]
    field_values.update(after_changes)
    return field_values


def _changed_values(
    source: WorkflowRow,
    comparison: WorkflowRow,
) -> dict[str, object]:
    return {
        key: value
        for key, value in source.items()
        if comparison.get(key) != value
    }


_ROW_IDENTITY_FIELDS = (
    "feature_id",
    "title",
    "kind",
    "assigned_role",
    "status",
)


def _json_object(text: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return dict(cast("Mapping[str, object]", parsed))


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
