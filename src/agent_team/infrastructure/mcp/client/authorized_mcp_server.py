"""Authorized MCP server wrapper."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from agents import AgentBase, RunContextWrapper
from agents.mcp import MCPServer
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    TextContent,
    Tool,
)

from agent_team.application.audit.audit_sanitizer import (
    sanitize_error,
    sanitize_tool_arguments,
    sanitize_tool_result,
)
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_denial import ToolInvocationDenial
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.capability_denied_error import (
    CapabilityDeniedError,
)
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.mcp.client.development_workflow_mcp_server_config import (  # noqa: E501
    DevelopmentWorkflowMCPServerConfig,
)

# Any is required by the installed Agents SDK MCPServer abstract methods.
# The installed MCP Tool schema uses dict[str, Any], so schema boundary casts
# are limited to this adapter.

CREATED_BY_ARGUMENT = "created_by"
CAPABILITY_DENIED_PREFIX = "CAPABILITY_DENIED"
NON_RETRYABLE_DENIAL_SUFFIX = (
    "Do not retry this action with different arguments."
)

READ_ONLY_TOOLS = frozenset(
    {
        WorkflowToolName.GET_FEATURE.value,
        WorkflowToolName.GET_FEATURE_OVERVIEW.value,
        WorkflowToolName.LIST_FEATURES.value,
        WorkflowToolName.LIST_ARTIFACTS.value,
        WorkflowToolName.LIST_TASKS.value,
    },
)

MUTATING_TOOLS = frozenset(
    {
        WorkflowToolName.CREATE_FEATURE.value,
        WorkflowToolName.ADD_ARTIFACT.value,
        WorkflowToolName.CREATE_TASK.value,
        WorkflowToolName.UPDATE_TASK_STATUS.value,
    },
)


class AuthorizedMCPServer(MCPServer):
    """MCP server wrapper that authorizes calls before delegation."""

    def __init__(
        self,
        delegate: MCPServer,
        config: DevelopmentWorkflowMCPServerConfig,
    ) -> None:
        """Create an authorized wrapper around an MCP server."""
        super().__init__(
            use_structured_content=delegate.use_structured_content,
            require_approval="never",
        )
        self.delegate = delegate
        self.profile = config.profile
        self.authorizer = config.authorizer
        self.audit_repository = config.audit_repository
        self.run = config.run
        self.bound_task_id = config.bound_task_id

    @property
    def name(self) -> str:
        """Return the readable server name."""
        return self.delegate.name

    @property
    def bound_feature_id(self) -> int | None:
        """Return the feature ID bound to this run, if any."""
        return self.run.feature_id

    @property
    def cached_tools(self) -> list[Tool] | None:
        """Return cached tools from the delegate server."""
        return self.delegate.cached_tools

    async def connect(self) -> None:
        """Connect the delegate server."""
        await self.delegate.connect()

    async def cleanup(self) -> None:
        """Clean up the delegate server."""
        await self.delegate.cleanup()

    async def list_tools(
        self,
        run_context: RunContextWrapper[Any] | None = None,
        agent: AgentBase | None = None,
    ) -> list[Tool]:
        """List only tools allowed by the active role profile."""
        tools = await self.delegate.list_tools(run_context, agent)
        allowed_names = {tool.value for tool in self.profile.allowed_tools}
        return [
            _model_visible_tool(tool)
            for tool in tools
            if tool.name in allowed_names
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Authorize and invoke a tool on the delegate server."""
        classification = _classify_tool(tool_name)
        try:
            self._reject_user_supplied_actor(tool_name, arguments)
            self.authorizer.authorize(
                self.profile,
                tool_name,
                arguments,
                bound_feature_id=self.run.feature_id,
                bound_task_id=self.bound_task_id,
            )
        except CapabilityDeniedError as error:
            self._record_denial(
                tool_name=tool_name,
                classification=classification,
                arguments=arguments,
                error=error,
            )
            return _capability_denied_result(error)

        delegated_arguments = self._delegated_arguments(
            tool_name,
            arguments,
        )
        arguments_hash, arguments_preview = sanitize_tool_arguments(
            tool_name,
            delegated_arguments,
        )
        invocation_start = ToolInvocationStart(
            run_id=self.run.id,
            server_name=self.name,
            tool_name=tool_name,
            classification=classification,
            arguments_hash=arguments_hash,
            arguments_preview_json=arguments_preview,
        )

        invocation = self.audit_repository.start_tool_invocation(
            invocation_start,
        )
        try:
            result = await self.delegate.call_tool(
                tool_name,
                delegated_arguments,
                meta,
            )
        except Exception as error:
            error_type, error_message = sanitize_error(error)
            try:
                self.audit_repository.fail_tool_invocation(
                    invocation_id=invocation.id,
                    error_type=error_type,
                    error_message=error_message,
                )
            except Exception as audit_error:
                raise audit_error from error
            raise

        result_hash, result_preview = sanitize_tool_result(tool_name, result)
        self.audit_repository.complete_tool_invocation(
            invocation_id=invocation.id,
            result_hash=result_hash,
            result_preview=result_preview,
        )
        return result

    def _record_denial(
        self,
        tool_name: str,
        classification: ToolClassification,
        arguments: dict[str, Any] | None,
        error: CapabilityDeniedError,
    ) -> None:
        error_type, error_message = sanitize_error(error)
        arguments_hash, arguments_preview = sanitize_tool_arguments(
            tool_name,
            arguments,
        )
        try:
            self.audit_repository.deny_tool_invocation(
                ToolInvocationDenial(
                    invocation=ToolInvocationStart(
                        run_id=self.run.id,
                        server_name=self.name,
                        tool_name=tool_name,
                        classification=classification,
                        arguments_hash=arguments_hash,
                        arguments_preview_json=arguments_preview,
                    ),
                    error_type=error_type,
                    error_message=error_message,
                ),
            )
        except Exception as audit_error:
            raise audit_error from error

    def _reject_user_supplied_actor(
        self,
        tool_name: str,
        arguments: Mapping[str, object] | None,
    ) -> None:
        if (
            tool_name == WorkflowToolName.ADD_ARTIFACT.value
            and arguments is not None
            and CREATED_BY_ARGUMENT in arguments
        ):
            raise CapabilityDeniedError(
                f"The {self.profile.role.value} role cannot provide "
                "created_by; actor identity is assigned by trusted runtime "
                "context.",
            )

    def _delegated_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if tool_name == WorkflowToolName.CREATE_TASK.value:
            delegated_arguments = dict(arguments or {})
            delegated_arguments.setdefault("status", TaskStatus.PENDING.value)
            return delegated_arguments

        if tool_name != WorkflowToolName.ADD_ARTIFACT.value:
            return arguments

        delegated_arguments = dict(arguments or {})
        delegated_arguments[CREATED_BY_ARGUMENT] = _trusted_actor(
            self.profile,
        )
        return delegated_arguments

    async def list_prompts(self) -> ListPromptsResult:
        """List prompts from the delegate server."""
        return await self.delegate.list_prompts()

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        """Get a prompt from the delegate server."""
        return await self.delegate.get_prompt(name, arguments)


def _classify_tool(tool_name: str) -> ToolClassification:
    if tool_name in READ_ONLY_TOOLS:
        return ToolClassification.READ_ONLY
    if tool_name in MUTATING_TOOLS:
        return ToolClassification.MUTATING
    return ToolClassification.PROHIBITED


def _model_visible_tool(tool: Tool) -> Tool:
    if tool.name != WorkflowToolName.ADD_ARTIFACT.value:
        return tool

    input_schema = _copy_mapping(
        cast("Mapping[str, object]", tool.input_schema),
    )
    properties = input_schema.get("properties")
    if isinstance(properties, dict):
        copied_properties = _copy_mapping(
            cast("Mapping[str, object]", properties),
        )
        copied_properties.pop(CREATED_BY_ARGUMENT, None)
        input_schema["properties"] = copied_properties

    required = input_schema.get("required")
    if isinstance(required, list):
        required_values = cast("Sequence[object]", required)
        input_schema["required"] = [
            value for value in required_values if value != CREATED_BY_ARGUMENT
        ]

    return tool.model_copy(update={"input_schema": input_schema}, deep=True)


def _copy_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {key: _copy_value(value) for key, value in values.items()}


def _copy_value(value: object) -> object:
    if isinstance(value, dict):
        return _copy_mapping(cast("Mapping[str, object]", value))
    if isinstance(value, list):
        values = cast("Sequence[object]", value)
        return [_copy_value(item) for item in values]
    return value


def _trusted_actor(profile: AgentProfile) -> str:
    return f"agent:{profile.role.value}"


def _capability_denied_result(
    error: CapabilityDeniedError,
) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(text=_capability_denied_message(error)),
        ],
        is_error=True,
    )


def _capability_denied_message(error: CapabilityDeniedError) -> str:
    detail = str(error) or "The requested workflow capability is not allowed."
    return (
        f"{CAPABILITY_DENIED_PREFIX}: {detail} {NON_RETRYABLE_DENIAL_SUFFIX}"
    )
