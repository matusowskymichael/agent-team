"""Role profile catalog for agent runs."""

from dataclasses import dataclass, field

from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_run_limits import AgentRunLimits
from agent_team.domain.runtime.capability_denied_error import (
    CapabilityDeniedError,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.skills.agent_skill_name import AgentSkillName
from agent_team.domain.workspace.workspace_tool_name import WorkspaceToolName

READ_WORKFLOW_TOOLS = frozenset(
    {
        WorkflowToolName.GET_FEATURE,
        WorkflowToolName.GET_FEATURE_OVERVIEW,
        WorkflowToolName.LIST_FEATURES,
        WorkflowToolName.LIST_ARTIFACTS,
        WorkflowToolName.LIST_TASKS,
    },
)
FEATURE_SCOPED_READ_WORKFLOW_TOOLS = READ_WORKFLOW_TOOLS - frozenset(
    {WorkflowToolName.LIST_FEATURES},
)

ALL_WORKFLOW_TOOLS = frozenset(WorkflowToolName)
DEFAULT_RUN_LIMITS = AgentRunLimits()
DEVELOPER_RUN_LIMITS = AgentRunLimits(max_turns=10)
BUSINESS_ANALYST_SKILLS = frozenset(
    {
        AgentSkillName("write-requirements-artifact"),
        AgentSkillName("write-acceptance-criteria"),
        AgentSkillName("review-feature-readiness"),
    },
)
SOFTWARE_ARCHITECT_SKILLS = frozenset(
    {
        AgentSkillName("review-architecture-readiness"),
        AgentSkillName("design-solution-architecture"),
        AgentSkillName("write-implementation-plan"),
        AgentSkillName("decompose-development-tasks"),
    },
)
BACKEND_DEVELOPER_SKILLS = frozenset(
    {
        AgentSkillName("implement-backend-task"),
    },
)
FRONTEND_DEVELOPER_SKILLS = frozenset(
    {
        AgentSkillName("implement-frontend-task"),
    },
)
DEVELOPER_WORKSPACE_TOOLS = frozenset(WorkspaceToolName)
BACKEND_WORKSPACE_PATH_PREFIXES = frozenset(
    {
        "api",
        "app",
        "backend",
        "server",
        "services",
        "shared",
        "src",
        "tests",
    },
)
FRONTEND_WORKSPACE_PATH_PREFIXES = frozenset(
    {
        "components",
        "frontend",
        "hooks",
        "pages",
        "shared",
        "src",
        "styles",
        "tests",
        "ui",
        "web",
    },
)
BACKEND_WORKSPACE_CHECKS = frozenset(
    {
        "backend",
        "pytest",
        "pyright",
        "ruff",
    },
)
FRONTEND_WORKSPACE_CHECKS = frozenset(
    {
        "frontend",
        "pytest",
        "ruff",
    },
)

BASE_INSTRUCTIONS = """
You are the local development workflow coordinator.
Use only available development workflow MCP tools when the user asks to
create, retrieve, list, or update workflow records.
Never claim workflow data was created or updated unless the relevant tool
call succeeded.
Never invent feature IDs or task IDs. Use IDs returned by tools.
If required information is missing, ask the user instead of guessing.
Summarize successful changes concisely and include returned IDs.
Your tool access is role-limited by code and cannot be expanded by prompts.
""".strip()

SOFTWARE_ARCHITECT_INSTRUCTIONS = f"""
{BASE_INSTRUCTIONS}

You are the Software Architect specialist.
Transform approved requirements and acceptance criteria into architecture
analysis, architecture artifacts, implementation-plan artifacts, and
actionable tasks for delivery specialists.
Review authoritative requirements and acceptance criteria before proposing
architecture. Identify blockers, contradictions, assumptions, tradeoffs, and
unresolved questions.
Ask targeted questions when material business decisions are missing. Do not
invent missing business requirements or silently resolve material business
ambiguities.
Preview architecture or implementation-plan content without saving when the
user asks to propose, draft, review, or not save.
Save artifacts or create tasks only when explicitly requested, and verify every
mutation from a successful tool result before claiming success.
The run/session bound feature ID is immutable. If the user requests another
feature ID, refuse without calling feature-scoped tools for that other feature
and tell the user to start a separate correctly bound session.
Do not write application source code or use filesystem, shell, network,
browser, package-manager, Git, or source-control tools.
""".strip()


def _default_profiles() -> dict[DevelopmentRole, AgentProfile]:
    return {
        role: AgentProfile(
            role=role,
            instructions=_instructions_for(role),
            allowed_tools=tools,
            run_limits=_role_run_limits().get(role, DEFAULT_RUN_LIMITS),
            allowed_skill_names=_role_skills().get(role, frozenset()),
            allowed_workspace_tools=_role_workspace_tools().get(
                role,
                frozenset(),
            ),
            allowed_workspace_path_prefixes=_role_path_prefixes().get(
                role,
                frozenset(),
            ),
            allowed_workspace_checks=_role_checks().get(role, frozenset()),
        )
        for role, tools in _role_tools().items()
    }


def _role_tools() -> dict[DevelopmentRole, frozenset[WorkflowToolName]]:
    return {
        DevelopmentRole.DELIVERY_MANAGER: ALL_WORKFLOW_TOOLS,
        DevelopmentRole.BUSINESS_ANALYST: READ_WORKFLOW_TOOLS
        | frozenset({WorkflowToolName.ADD_ARTIFACT}),
        DevelopmentRole.SOFTWARE_ARCHITECT: FEATURE_SCOPED_READ_WORKFLOW_TOOLS
        | frozenset(
            {
                WorkflowToolName.ADD_ARTIFACT,
                WorkflowToolName.CREATE_TASK,
            },
        ),
        DevelopmentRole.BACKEND_DEVELOPER: READ_WORKFLOW_TOOLS
        | frozenset({WorkflowToolName.UPDATE_TASK_STATUS}),
        DevelopmentRole.FRONTEND_DEVELOPER: READ_WORKFLOW_TOOLS
        | frozenset({WorkflowToolName.UPDATE_TASK_STATUS}),
        DevelopmentRole.QA_ENGINEER: READ_WORKFLOW_TOOLS
        | frozenset(
            {
                WorkflowToolName.ADD_ARTIFACT,
                WorkflowToolName.UPDATE_TASK_STATUS,
            },
        ),
        DevelopmentRole.CODE_REVIEWER: READ_WORKFLOW_TOOLS
        | frozenset(
            {
                WorkflowToolName.ADD_ARTIFACT,
                WorkflowToolName.UPDATE_TASK_STATUS,
            },
        ),
    }


def _role_skills() -> dict[DevelopmentRole, frozenset[AgentSkillName]]:
    return {
        DevelopmentRole.BUSINESS_ANALYST: BUSINESS_ANALYST_SKILLS,
        DevelopmentRole.SOFTWARE_ARCHITECT: SOFTWARE_ARCHITECT_SKILLS,
        DevelopmentRole.BACKEND_DEVELOPER: BACKEND_DEVELOPER_SKILLS,
        DevelopmentRole.FRONTEND_DEVELOPER: FRONTEND_DEVELOPER_SKILLS,
    }


def _role_workspace_tools() -> dict[
    DevelopmentRole,
    frozenset[WorkspaceToolName],
]:
    return {
        DevelopmentRole.BACKEND_DEVELOPER: DEVELOPER_WORKSPACE_TOOLS,
        DevelopmentRole.FRONTEND_DEVELOPER: DEVELOPER_WORKSPACE_TOOLS,
    }


def _role_path_prefixes() -> dict[DevelopmentRole, frozenset[str]]:
    return {
        DevelopmentRole.BACKEND_DEVELOPER: BACKEND_WORKSPACE_PATH_PREFIXES,
        DevelopmentRole.FRONTEND_DEVELOPER: FRONTEND_WORKSPACE_PATH_PREFIXES,
    }


def _role_checks() -> dict[DevelopmentRole, frozenset[str]]:
    return {
        DevelopmentRole.BACKEND_DEVELOPER: BACKEND_WORKSPACE_CHECKS,
        DevelopmentRole.FRONTEND_DEVELOPER: FRONTEND_WORKSPACE_CHECKS,
    }


def _role_run_limits() -> dict[DevelopmentRole, AgentRunLimits]:
    return {
        DevelopmentRole.BACKEND_DEVELOPER: DEVELOPER_RUN_LIMITS,
        DevelopmentRole.FRONTEND_DEVELOPER: DEVELOPER_RUN_LIMITS,
    }


def _instructions_for(role: DevelopmentRole) -> str:
    if role is DevelopmentRole.SOFTWARE_ARCHITECT:
        return SOFTWARE_ARCHITECT_INSTRUCTIONS
    if role is DevelopmentRole.BACKEND_DEVELOPER:
        return f"{BASE_INSTRUCTIONS}\n\n{_backend_developer_instructions()}"
    if role is DevelopmentRole.FRONTEND_DEVELOPER:
        return f"{BASE_INSTRUCTIONS}\n\n{_frontend_developer_instructions()}"
    return BASE_INSTRUCTIONS


def _backend_developer_instructions() -> str:
    return """
You are the Backend Developer specialist.
Work only on the trusted assigned backend development task. Read the assigned
task and relevant requirements, acceptance criteria, architecture, and
implementation-plan artifacts before editing code.
Inspect the workspace structure, search proposed symbol names, search related
behavior, and read plausible matches before creating classes, functions,
methods, endpoints, models, repositories, services, or utilities.
Prefer reusing or extending existing backend or shared implementations. If new
code is necessary, state briefly why existing code could not be reused.
Modify only authorized backend or explicitly shared workspace paths. Refuse
frontend-only, cross-feature, cross-task, or unassigned-task requests without
attempting forbidden calls.
After editing, prefer the aggregate configured backend check
run_check(name="backend") because it covers the normal backend verification
suite. Use individual ruff, pyright, or pytest checks only when requested or
diagnostically necessary. Report changed files, reused code, checks, and
limitations truthfully.
""".strip()


def _frontend_developer_instructions() -> str:
    return """
You are the Frontend Developer specialist.
Work only on the trusted assigned frontend development task. Read the assigned
task and relevant requirements, acceptance criteria, architecture, and
implementation-plan artifacts before editing code.
Inspect existing components, styles, hooks, utilities, and workspace structure
before creating new UI code. Search proposed component or utility names and
related behavior, then read plausible matches.
Prefer reusing or extending existing frontend or shared implementations. If new
code is necessary, state briefly why existing code could not be reused.
Modify only authorized frontend or explicitly shared workspace paths. Refuse
backend-only, cross-feature, cross-task, or unassigned-task mutations.
After editing, prefer the aggregate configured frontend check
run_check(name="frontend") because it covers the normal frontend verification
suite. Use individual ruff or pytest checks only when requested or
diagnostically necessary. Report changed files, reused code, checks, and
limitations truthfully.
""".strip()


@dataclass(frozen=True, slots=True)
class AgentProfileCatalog:
    """Catalog of immutable role profiles."""

    profiles: dict[DevelopmentRole, AgentProfile] = field(
        default_factory=_default_profiles,
    )

    def get_profile(self, role: DevelopmentRole) -> AgentProfile:
        """Return the immutable profile for a development role."""
        profile = self.profiles.get(role)
        if profile is None:
            raise CapabilityDeniedError("Capability denied.")
        return profile
