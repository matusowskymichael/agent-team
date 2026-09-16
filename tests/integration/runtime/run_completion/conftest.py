"""SQLite and restricted workspace setup for logical run integration tests."""

import sys
from pathlib import Path

import pytest

from agent_team.application.context.feature_context_builder import (
    FeatureContextBuilder,
)
from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.application.sessions.agent_session_service import (
    AgentSessionService,
)
from agent_team.application.workflow.task_verification_service import (
    TaskVerificationService,
)
from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.application.workspace.workspace_service import WorkspaceService
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_agent_session_repository as session_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)
from tests.integration.runtime.run_completion.completion_operations import (
    CompletionOperations,
)
from tests.integration.runtime.run_completion.completion_runtime import (
    ScriptedCompletionRuntime,
)
from tests.integration.runtime.run_completion.completion_verifier import (
    CompletionVerifier,
)
from tests.integration.runtime.run_completion.run_completion_scenario import (
    RunCompletionScenario,
)


@pytest.fixture
def run_completion_scenario(tmp_path: Path) -> RunCompletionScenario:
    """Bind one pending backend task to real local storage and workspace."""
    database_path = tmp_path / "completion.db"
    workspace_root = tmp_path / "workspace"
    backend = workspace_root / "backend"
    backend.mkdir(parents=True)
    (backend / "auth.py").write_text(
        "class AuthService:\n    def logout(self):\n        return 0\n",
        encoding="utf-8",
    )
    for index in range(40):
        (backend / f"discovery_{index}.py").write_text(
            f"DISCOVERED_VALUE = {index}\n", encoding="utf-8"
        )
    repository = workflow_repository_module.SQLiteWorkflowRepository(
        database_path
    )
    workflow = WorkflowService(repository)
    feature = workflow.create_feature("Logout", "Implement the bound task.")
    workflow.add_artifact(
        feature_id=feature.id,
        kind=ArtifactKind.REQUIREMENTS,
        content="Revise the existing AuthService.logout method.",
        created_by="agent:business_analyst",
    )
    development_task = workflow.create_task(
        feature.id,
        "Revise logout",
        "Implement the requested revision.",
        DevelopmentRole.BACKEND_DEVELOPER,
    )
    audit = audit_repository_module.SQLiteAgentAuditRepository(database_path)
    sessions = session_repository_module.SQLiteAgentSessionRepository(
        database_path
    )
    operations = CompletionOperations(
        workflow=workflow,
        workspace=WorkspaceService(
            repository=repository,
            executor=LocalWorkspaceExecutor(
                root=workspace_root,
                check_commands={"backend": (sys.executable, "-c", "pass")},
            ),
        ),
        audit=audit,
    )
    runtime = ScriptedCompletionRuntime(operations=operations)
    verifier = CompletionVerifier()
    harness = AgentHarness(
        runtime=runtime,
        audit_repository=audit,
        workflow_repository=repository,
        session_service=AgentSessionService(
            repository=sessions, workflow_repository=repository
        ),
        context_provider=FeatureContextBuilder(repository),
        task_verification_service=TaskVerificationService(
            repository=repository, verifier=verifier, audit_reader=audit
        ),
    )
    return RunCompletionScenario(
        harness=harness,
        task=AgentTask(
            prompt="Implement the assigned logout revision to completion.",
            role=DevelopmentRole.BACKEND_DEVELOPER,
            feature_id=feature.id,
            task_id=development_task.id,
            workspace_root=workspace_root,
        ),
        runtime=runtime,
        verifier=verifier,
        repository=repository,
        audit=audit,
        sessions=sessions,
    )
