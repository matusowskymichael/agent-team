"""Agent role profile."""

from dataclasses import dataclass, field

from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.domain.workspace.workspace_tool_name import WorkspaceToolName


def _empty_skill_names() -> frozenset[AgentSkillName]:
    return frozenset()


def _empty_workspace_tool_names() -> frozenset[WorkspaceToolName]:
    return frozenset()


def _empty_path_prefixes() -> frozenset[str]:
    return frozenset()


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Immutable runtime profile for one development role."""

    role: DevelopmentRole
    instructions: str
    allowed_tools: frozenset[WorkflowToolName]
    run_limits: AgentRunLimits
    allowed_skill_names: frozenset[AgentSkillName] = field(
        default_factory=_empty_skill_names,
    )
    allowed_workspace_tools: frozenset[WorkspaceToolName] = field(
        default_factory=_empty_workspace_tool_names,
    )
    allowed_workspace_path_prefixes: frozenset[str] = field(
        default_factory=_empty_path_prefixes,
    )
    allowed_workspace_checks: frozenset[str] = field(
        default_factory=_empty_path_prefixes,
    )
