"""Command-line interface for the agent team application."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from agent_team.application.context.feature_context_builder import (
    FeatureContextBuilder,
)
from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
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
from agent_team.domain.context.agent_context_budget_exceeded_error import (
    AgentContextBudgetExceededError,
)
from agent_team.domain.runtime.agent_output_blank_error import (
    AgentOutputBlankError,
)
from agent_team.domain.runtime.agent_output_incomplete_error import (
    AgentOutputIncompleteError,
)
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.capability_denied_error import (
    CapabilityDeniedError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.sessions.agent_session_binding_error import (
    AgentSessionBindingError,
)
from agent_team.domain.sessions.invalid_agent_session_id_error import (
    InvalidAgentSessionIdError,
)
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)
from agent_team.domain.workflow.feature_not_found_error import (
    FeatureNotFoundError,
)
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_binding_error import (
    WorkspaceBindingError,
)
from agent_team.infrastructure.configuration.workflow_database_path import (
    load_workflow_database_path,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_config as workflow_mcp_config,
)
from agent_team.infrastructure.mcp.client import (
    development_workflow_mcp_server_factory as workflow_mcp_factory,
)
from agent_team.infrastructure.ollama.ollama_agent_executor import (
    OllamaAgentExecutor,
)
from agent_team.infrastructure.ollama.ollama_model_capability_error import (
    OllamaModelCapabilityError,
)
from agent_team.infrastructure.ollama.ollama_model_catalog import (
    ensure_ollama_model_ready,
    list_installed_ollama_models,
)
from agent_team.infrastructure.ollama.ollama_model_factory import (
    create_ollama_model,
)
from agent_team.infrastructure.ollama.ollama_model_unavailable_error import (
    OllamaModelUnavailableError,
)
from agent_team.infrastructure.ollama.ollama_settings import (
    OllamaSettings,
    load_ollama_settings,
)
from agent_team.infrastructure.ollama.ollama_unavailable_error import (
    OllamaUnavailableError,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_audit_migration_error,
)
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_agent_session_repository as session_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_session_factory as session_factory_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)
from agent_team.infrastructure.skills.agent_skill_tool_factory import (
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

from ...infrastructure.mcp.client.workflow_mcp_unavailable_error import (
    WorkflowMCPUnavailableError,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    arguments = _parse_arguments(argv)

    try:
        if arguments.list_models:
            _print_installed_models(cast("str | None", arguments.model))
            return 0
        if arguments.list_skills:
            _print_available_skills(
                DevelopmentRole(cast("str", arguments.role)),
            )
            return 0

        prompt = cast("str", arguments.prompt)
        task = AgentTask(
            prompt=prompt,
            role=DevelopmentRole(cast("str", arguments.role)),
            feature_id=cast("int | None", arguments.feature_id),
            session_id=cast("str | None", arguments.session_id),
            task_id=cast("int | None", arguments.task_id),
            workspace_root=cast("Path | None", arguments.workspace_root),
        )
        result = asyncio.run(
            run_prompt(
                task,
                model=cast("str | None", arguments.model),
            ),
        )
    except (
        AgentContextBudgetExceededError,
        AgentSessionBindingError,
        CapabilityDeniedError,
        FeatureNotFoundError,
        InvalidAgentSessionIdError,
        InvalidAgentSkillError,
        AgentOutputBlankError,
        AgentOutputIncompleteError,
        OllamaModelCapabilityError,
        OllamaModelUnavailableError,
        OllamaUnavailableError,
        sqlite_audit_migration_error.SQLiteAuditMigrationError,
        WorkflowMCPUnavailableError,
        WorkspaceAccessDeniedError,
        WorkspaceBindingError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(result.response)
    return 0


def build_orchestrator(settings: OllamaSettings | None = None) -> Orchestrator:
    """Build the application orchestrator with local infrastructure."""
    ollama_settings = load_ollama_settings() if settings is None else settings
    model = create_ollama_model(ollama_settings)
    database_path = load_workflow_database_path()
    workflow_repository = workflow_repository_module.SQLiteWorkflowRepository(
        database_path,
    )
    audit_repository = audit_repository_module.SQLiteAgentAuditRepository(
        database_path,
    )
    session_repository = (
        session_repository_module.SQLiteAgentSessionRepository(database_path)
    )
    session_factory = session_factory_module.SQLiteSessionFactory(
        database_path,
    )
    authorizer = CapabilityAuthorizer(repository=workflow_repository)
    skill_service = build_skill_service()
    skill_tool_factory = AgentSkillToolFactory(
        service=skill_service,
        audit_repository=audit_repository,
    )
    workspace_tool_factory = WorkspaceToolFactory(
        service_factory=lambda workspace_root: WorkspaceService(
            repository=workflow_repository,
            executor=LocalWorkspaceExecutor(workspace_root),
        ),
        audit_repository=audit_repository,
    )
    runtime = OllamaAgentExecutor(
        model=model,
        settings=ollama_settings,
        mcp_server_factory=lambda profile, run, task: (
            workflow_mcp_factory.create_development_workflow_mcp_server(
                workflow_mcp_config.DevelopmentWorkflowMCPServerConfig(
                    profile=profile,
                    authorizer=authorizer,
                    audit_repository=audit_repository,
                    run=run,
                    bound_task_id=task.task_id,
                ),
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
            context_provider=FeatureContextBuilder(workflow_repository),
            skill_service=skill_service,
        ),
    )


def build_skill_service() -> AgentSkillService:
    """Build read-only local Agent Skill services."""
    return AgentSkillService(
        catalog=FilesystemAgentSkillCatalog(),
        authorizer=AgentSkillAuthorizer(),
    )


def _parse_arguments(
    argv: Sequence[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-team")
    parser.add_argument(
        "--role",
        choices=[role.value for role in DevelopmentRole],
        default=DevelopmentRole.DELIVERY_MANAGER.value,
    )
    parser.add_argument("--feature-id", type=_positive_integer)
    parser.add_argument("--session-id")
    parser.add_argument("--task-id", type=_positive_integer)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List locally installed Ollama models and exit.",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List available local Agent Skills for the selected role.",
    )
    parser.add_argument("prompt", nargs="?")
    namespace = parser.parse_args(argv)
    if (
        not namespace.list_models
        and not namespace.list_skills
        and namespace.prompt is None
    ):
        parser.error(
            "prompt is required unless --list-models or --list-skills is used",
        )
    return namespace


async def run_prompt(
    task: AgentTask,
    model: str | None = None,
) -> AgentResult:
    """Run a prompt through the composed application orchestrator."""
    settings = load_ollama_settings(model_override=model)
    ensure_ollama_model_ready(settings)
    orchestrator = build_orchestrator(settings)
    return await orchestrator.run(task)


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


def _print_installed_models(model: str | None) -> None:
    settings = load_ollama_settings(model_override=model)
    for installed_model in list_installed_ollama_models(settings):
        print(installed_model)


def _print_available_skills(role: DevelopmentRole) -> None:
    profile = AgentProfileCatalog().get_profile(role)
    service = build_skill_service()
    metadata = service.list_available_metadata(profile)
    if not metadata:
        print("No skills available.")
        return
    print("NAME\tVERSION\tHASH\tDESCRIPTION")
    for skill in sorted(metadata, key=lambda item: item.name.value):
        print(
            "\t".join(
                (
                    skill.name.value,
                    skill.version or "-",
                    skill.content_hash[:12],
                    skill.description,
                ),
            ),
        )
