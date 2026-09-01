"""Configuration for authorized workflow MCP server construction."""

from dataclasses import dataclass

from agent_team.application.runtime.capability_authorizer import (
    CapabilityAuthorizer,
)
from agent_team.domain.audit.agent_audit_repository import AgentAuditRepository
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.runtime.agent_profile import AgentProfile


@dataclass(frozen=True, slots=True)
class DevelopmentWorkflowMCPServerConfig:
    """Dependencies and trusted context for workflow MCP authorization."""

    profile: AgentProfile
    authorizer: CapabilityAuthorizer
    audit_repository: AgentAuditRepository
    run: AgentRunRecord
    bound_task_id: int | None = None
