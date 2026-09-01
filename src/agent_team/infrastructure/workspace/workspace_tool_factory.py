"""Agents SDK function tools for restricted workspace access."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from agents import FunctionTool, Tool
from agents.tool_context import ToolContext

from agent_team.application.audit.audit_sanitizer import (
    sanitize_error,
    sanitize_tool_arguments,
    sanitize_tool_result,
)
from agent_team.application.workspace.workspace_service import (
    WorkspaceService,
)
from agent_team.domain.audit.agent_audit_repository import AgentAuditRepository
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_denial import (
    ToolInvocationDenial,
)
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.workspace.check_run_result import CheckRunResult
from agent_team.domain.workspace.code_search_result import CodeSearchResult
from agent_team.domain.workspace.patch_application_result import (
    PatchApplicationResult,
)
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)
from agent_team.domain.workspace.workspace_binding_error import (
    WorkspaceBindingError,
)
from agent_team.domain.workspace.workspace_file_content import (
    WorkspaceFileContent,
)
from agent_team.domain.workspace.workspace_file_listing import (
    WorkspaceFileListing,
)
from agent_team.domain.workspace.workspace_tool_name import WorkspaceToolName

# Any is required by the installed Agents SDK FunctionTool callable boundary.

WORKSPACE_SERVER_NAME = "workspace"
WORKSPACE_DENIED_PREFIX = "WORKSPACE_CAPABILITY_DENIED"
WORKSPACE_DENIED_SUFFIX = (
    "Do not retry with different feature, task, role, or workspace values."
)
READ_ONLY_WORKSPACE_TOOLS = frozenset(
    {
        WorkspaceToolName.LIST_FILES,
        WorkspaceToolName.SEARCH_CODE,
        WorkspaceToolName.READ_FILE,
        WorkspaceToolName.RUN_CHECK,
    },
)


@dataclass(frozen=True, slots=True)
class WorkspaceToolFactory:
    """Build restricted workspace function tools for one agent run."""

    service_factory: Callable[[Path], WorkspaceService]
    audit_repository: AgentAuditRepository

    def create_tools(
        self,
        profile: AgentProfile,
        run: AgentRunRecord,
        task: AgentTask,
    ) -> list[Tool]:
        """Return SDK function tools allowed by the active profile."""
        if not profile.allowed_workspace_tools:
            return []

        async def list_files(
            _context: ToolContext[Any],
            arguments_json: str,
        ) -> dict[str, object]:
            return self._invoke(
                profile,
                run,
                task,
                WorkspaceToolName.LIST_FILES,
                arguments_json,
            )

        async def search_code(
            _context: ToolContext[Any],
            arguments_json: str,
        ) -> dict[str, object]:
            return self._invoke(
                profile,
                run,
                task,
                WorkspaceToolName.SEARCH_CODE,
                arguments_json,
            )

        async def read_file(
            _context: ToolContext[Any],
            arguments_json: str,
        ) -> dict[str, object]:
            return self._invoke(
                profile,
                run,
                task,
                WorkspaceToolName.READ_FILE,
                arguments_json,
            )

        async def apply_patch(
            _context: ToolContext[Any],
            arguments_json: str,
        ) -> dict[str, object]:
            return self._invoke(
                profile,
                run,
                task,
                WorkspaceToolName.APPLY_PATCH,
                arguments_json,
            )

        async def run_check(
            _context: ToolContext[Any],
            arguments_json: str,
        ) -> dict[str, object]:
            return self._invoke(
                profile,
                run,
                task,
                WorkspaceToolName.RUN_CHECK,
                arguments_json,
            )

        callbacks = {
            WorkspaceToolName.LIST_FILES: list_files,
            WorkspaceToolName.SEARCH_CODE: search_code,
            WorkspaceToolName.READ_FILE: read_file,
            WorkspaceToolName.APPLY_PATCH: apply_patch,
            WorkspaceToolName.RUN_CHECK: run_check,
        }
        return [
            FunctionTool(
                name=tool.value,
                description=_tool_description(tool),
                params_json_schema=_tool_schema(tool, profile),
                on_invoke_tool=callbacks[tool],
                strict_json_schema=True,
            )
            for tool in sorted(
                profile.allowed_workspace_tools,
                key=lambda item: item.value,
            )
        ]

    def _invoke(
        self,
        profile: AgentProfile,
        run: AgentRunRecord,
        task: AgentTask,
        tool: WorkspaceToolName,
        arguments_json: str,
    ) -> dict[str, object]:
        arguments = _arguments(arguments_json)
        if task.workspace_root is None:
            return self._deny(
                run.id,
                tool,
                arguments,
                WorkspaceBindingError(
                    "Workspace tools require a trusted workspace root.",
                ),
            )
        classification = _classification(tool)
        try:
            service = self.service_factory(task.workspace_root)
            service.authorize(
                profile,
                task,
                tool,
                _path_argument(tool, arguments),
                classification is ToolClassification.MUTATING,
            )
            _authorize_check_name(profile, tool, arguments)
        except (WorkspaceAccessDeniedError, WorkspaceBindingError) as error:
            return self._deny(run.id, tool, arguments, error)

        invocation = self._start_invocation(
            run_id=run.id,
            tool=tool,
            classification=classification,
            arguments=arguments,
        )
        try:
            result = _call_service(service, profile, task, tool, arguments)
        except (WorkspaceAccessDeniedError, WorkspaceBindingError) as error:
            self._mark_denied(invocation.id, error)
            return _denied_response(error)
        except Exception as error:
            self._mark_failed(invocation.id, error)
            return _failed_response(error)

        payload = _result_payload(result)
        result_hash, result_preview = sanitize_tool_result(tool.value, payload)
        self.audit_repository.complete_tool_invocation(
            invocation_id=invocation.id,
            result_hash=result_hash,
            result_preview=result_preview,
        )
        return payload

    def _deny(
        self,
        run_id: int,
        tool: WorkspaceToolName,
        arguments: dict[str, object],
        error: WorkspaceAccessDeniedError | WorkspaceBindingError,
    ) -> dict[str, object]:
        arguments_hash, arguments_preview = sanitize_tool_arguments(
            tool.value,
            arguments,
        )
        error_type, error_message = sanitize_error(error)
        self.audit_repository.deny_tool_invocation(
            ToolInvocationDenial(
                invocation=ToolInvocationStart(
                    run_id=run_id,
                    server_name=WORKSPACE_SERVER_NAME,
                    tool_name=tool.value,
                    classification=_classification(tool),
                    arguments_hash=arguments_hash,
                    arguments_preview_json=arguments_preview,
                ),
                error_type=error_type,
                error_message=error_message,
            ),
        )
        return _denied_response(error)

    def _start_invocation(
        self,
        run_id: int,
        tool: WorkspaceToolName,
        classification: ToolClassification,
        arguments: dict[str, object],
    ) -> ToolInvocationRecord:
        arguments_hash, arguments_preview = sanitize_tool_arguments(
            tool.value,
            arguments,
        )
        return self.audit_repository.start_tool_invocation(
            ToolInvocationStart(
                run_id=run_id,
                server_name=WORKSPACE_SERVER_NAME,
                tool_name=tool.value,
                classification=classification,
                arguments_hash=arguments_hash,
                arguments_preview_json=arguments_preview,
            ),
        )

    def _mark_denied(
        self,
        invocation_id: int,
        error: WorkspaceAccessDeniedError | WorkspaceBindingError,
    ) -> None:
        error_type, error_message = sanitize_error(error)
        self.audit_repository.fail_tool_invocation(
            invocation_id=invocation_id,
            error_type=error_type,
            error_message=error_message,
        )

    def _mark_failed(self, invocation_id: int, error: Exception) -> None:
        error_type, error_message = sanitize_error(error)
        self.audit_repository.fail_tool_invocation(
            invocation_id=invocation_id,
            error_type=error_type,
            error_message=error_message,
        )


def _call_service(
    service: WorkspaceService,
    profile: AgentProfile,
    task: AgentTask,
    tool: WorkspaceToolName,
    arguments: dict[str, object],
) -> object:
    if tool is WorkspaceToolName.LIST_FILES:
        return service.list_files(
            profile,
            task,
            _optional_text(arguments, "directory"),
        )
    if tool is WorkspaceToolName.SEARCH_CODE:
        return service.search_code(
            profile,
            task,
            _required_text(arguments, "query"),
        )
    if tool is WorkspaceToolName.READ_FILE:
        return service.read_file(
            profile,
            task,
            _required_text(arguments, "path"),
        )
    if tool is WorkspaceToolName.APPLY_PATCH:
        return service.apply_patch(
            profile,
            task,
            _required_text(arguments, "path"),
            _required_text(arguments, "old_text"),
            _required_text(arguments, "new_text"),
        )
    if tool is WorkspaceToolName.RUN_CHECK:
        return service.run_check(
            profile,
            task,
            _required_text(arguments, "name"),
        )
    raise WorkspaceAccessDeniedError("Unsupported workspace tool.")


def _path_argument(
    tool: WorkspaceToolName,
    arguments: dict[str, object],
) -> str:
    if tool is WorkspaceToolName.LIST_FILES:
        return _optional_text(arguments, "directory")
    if tool in {WorkspaceToolName.READ_FILE, WorkspaceToolName.APPLY_PATCH}:
        return _required_text(arguments, "path")
    return ""


def _authorize_check_name(
    profile: AgentProfile,
    tool: WorkspaceToolName,
    arguments: dict[str, object],
) -> None:
    if tool is not WorkspaceToolName.RUN_CHECK:
        return
    name = _required_text(arguments, "name")
    if name not in profile.allowed_workspace_checks:
        raise WorkspaceAccessDeniedError(
            f"The {profile.role.value} role cannot run check {name}.",
        )


def _arguments(arguments_json: str) -> dict[str, object]:
    parsed = json.loads(arguments_json)
    if not isinstance(parsed, dict):
        raise WorkspaceAccessDeniedError("Tool arguments must be an object.")
    return dict(cast("dict[str, object]", parsed))


def _required_text(arguments: dict[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise WorkspaceAccessDeniedError(f"Tool argument {key} is required.")
    return value


def _optional_text(arguments: dict[str, object], key: str) -> str:
    value = arguments.get(key, "")
    if not isinstance(value, str):
        raise WorkspaceAccessDeniedError(f"Tool argument {key} must be text.")
    return value


def _result_payload(result: object) -> dict[str, object]:
    if isinstance(result, WorkspaceFileListing):
        return {"files": list(result.files), "truncated": result.truncated}
    if isinstance(result, CodeSearchResult):
        return {
            "query_hash": result.query_hash,
            "matches": [
                {
                    "path": match.path,
                    "line_number": match.line_number,
                    "line_excerpt": match.line_excerpt,
                }
                for match in result.matches
            ],
            "truncated": result.truncated,
        }
    if isinstance(result, WorkspaceFileContent):
        return {
            "path": result.path,
            "content": result.content,
            "content_hash": result.content_hash,
            "truncated": result.truncated,
        }
    if isinstance(result, PatchApplicationResult):
        return {
            "path": result.path,
            "applied": result.applied,
            "before_hash": result.before_hash,
            "after_hash": result.after_hash,
            "line_count_delta": result.line_count_delta,
            "message": result.message,
        }
    if isinstance(result, CheckRunResult):
        return {
            "name": result.name,
            "exit_code": result.exit_code,
            "stdout_excerpt": result.stdout_excerpt,
            "stderr_excerpt": result.stderr_excerpt,
            "timed_out": result.timed_out,
        }
    raise WorkspaceAccessDeniedError("Unsupported workspace tool result.")


def _classification(tool: WorkspaceToolName) -> ToolClassification:
    if tool in READ_ONLY_WORKSPACE_TOOLS:
        return ToolClassification.READ_ONLY
    return ToolClassification.MUTATING


def _denied_response(
    error: WorkspaceAccessDeniedError | WorkspaceBindingError,
) -> dict[str, object]:
    error_type, error_message = sanitize_error(error)
    return {
        "error": (
            f"{WORKSPACE_DENIED_PREFIX}: {error_message}. "
            f"{WORKSPACE_DENIED_SUFFIX}"
        ),
        "error_type": error_type,
    }


def _failed_response(error: Exception) -> dict[str, object]:
    error_type, error_message = sanitize_error(error)
    return {
        "error": f"WORKSPACE_TOOL_FAILED: {error_message}",
        "error_type": error_type,
    }


def _tool_schema(
    tool: WorkspaceToolName,
    profile: AgentProfile,
) -> dict[str, object]:
    if tool is WorkspaceToolName.LIST_FILES:
        return _object_schema(
            {
                "directory": {
                    "type": "string",
                    "description": (
                        "Workspace-relative directory to list; use an empty "
                        "string for the workspace root."
                    ),
                },
            },
            ["directory"],
        )
    if tool is WorkspaceToolName.SEARCH_CODE:
        return _object_schema(
            {
                "query": {
                    "type": "string",
                    "description": (
                        "Literal code text or symbol name to search for."
                    ),
                },
            },
            ["query"],
        )
    if tool is WorkspaceToolName.READ_FILE:
        return _object_schema(
            {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative file path to read. Absolute "
                        "paths and traversal are rejected."
                    ),
                },
            },
            ["path"],
        )
    if tool is WorkspaceToolName.APPLY_PATCH:
        return _object_schema(
            {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative file path to modify or create."
                    ),
                },
                "old_text": {
                    "type": "string",
                    "description": (
                        "Exact existing text to replace. Use an empty string "
                        "only when creating a new file."
                    ),
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text or new file content.",
                },
            },
            ["path", "old_text", "new_text"],
        )
    if tool is WorkspaceToolName.RUN_CHECK:
        return _object_schema(
            {
                "name": {
                    "type": "string",
                    "enum": sorted(profile.allowed_workspace_checks),
                    "description": "Name of an allowlisted project check.",
                },
            },
            ["name"],
        )
    raise WorkspaceAccessDeniedError("Unsupported workspace tool.")


def _object_schema(
    properties: dict[str, object],
    required: list[str],
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _tool_description(tool: WorkspaceToolName) -> str:
    descriptions = {
        WorkspaceToolName.LIST_FILES: (
            "List visible files below the trusted workspace root. Use before "
            "editing to understand repository structure."
        ),
        WorkspaceToolName.SEARCH_CODE: (
            "Search visible workspace files for literal code, symbol names, "
            "or related behavior before creating new implementation elements."
        ),
        WorkspaceToolName.READ_FILE: (
            "Read bounded content from one visible workspace file after "
            "listing or searching plausible matches."
        ),
        WorkspaceToolName.APPLY_PATCH: (
            "Apply the smallest exact text replacement inside authorized "
            "workspace paths. This cannot access files outside the trusted "
            "workspace root."
        ),
        WorkspaceToolName.RUN_CHECK: (
            "Run one configured project check by name. This is not arbitrary "
            "shell access."
        ),
    }
    return descriptions[tool]
