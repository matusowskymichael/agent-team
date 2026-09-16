"""Composite integration fixture with real repositories and fake inference."""

from dataclasses import dataclass

from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_agent_session_repository as session_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)
from tests.integration.runtime.run_completion.completion_runtime import (
    ScriptedCompletionRuntime,
)
from tests.integration.runtime.run_completion.completion_verifier import (
    CompletionVerifier,
)


@dataclass(frozen=True, slots=True)
class RunCompletionScenario:
    """Expose the trusted fixture and resulting persistence for assertions."""

    harness: AgentHarness
    task: AgentTask
    runtime: ScriptedCompletionRuntime
    verifier: CompletionVerifier
    repository: workflow_repository_module.SQLiteWorkflowRepository
    audit: audit_repository_module.SQLiteAgentAuditRepository
    sessions: session_repository_module.SQLiteAgentSessionRepository
