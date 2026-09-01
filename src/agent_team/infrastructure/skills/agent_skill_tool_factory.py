"""Agents SDK function tools for local Agent Skills."""

import json
from dataclasses import dataclass
from typing import Any, cast

from agents import FunctionTool, Tool
from agents.tool_context import ToolContext

from agent_team.application.audit.audit_sanitizer import (
    sanitize_error,
    sanitize_tool_arguments,
)
from agent_team.application.skills.agent_skill_service import (
    AgentSkillService,
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
from agent_team.domain.skills.agent_skill import AgentSkill
from agent_team.domain.skills.agent_skill_access_denied_error import (
    AgentSkillAccessDeniedError,
)
from agent_team.domain.skills.invalid_agent_skill_error import (
    InvalidAgentSkillError,
)

# Any is required by the installed Agents SDK FunctionTool callable boundary.

SKILL_SERVER_NAME = "agent_skills"
LOAD_SKILL_TOOL_NAME = "load_skill"
READ_SKILL_RESOURCE_TOOL_NAME = "read_skill_resource"
SKILL_ACCESS_DENIED_PREFIX = "SKILL_ACCESS_DENIED"


@dataclass(frozen=True, slots=True)
class AgentSkillToolFactory:
    """Build read-only Agent Skill tools for one agent run."""

    service: AgentSkillService
    audit_repository: AgentAuditRepository

    def create_tools(
        self,
        profile: AgentProfile,
        run: AgentRunRecord,
    ) -> list[Tool]:
        """Return SDK function tools when the profile has visible skills."""
        if not profile.allowed_skill_names:
            return []

        async def load_skill(
            _context: ToolContext[Any],
            arguments_json: str,
        ) -> dict[str, object]:
            return self._load_skill(profile, run.id, arguments_json)

        async def read_skill_resource(
            _context: ToolContext[Any],
            arguments_json: str,
        ) -> dict[str, object]:
            return self._read_skill_resource(profile, run.id, arguments_json)

        return [
            FunctionTool(
                name=LOAD_SKILL_TOOL_NAME,
                description=(
                    "Load the full instructions for one available local "
                    "Agent Skill by name."
                ),
                params_json_schema=_load_skill_schema(),
                on_invoke_tool=load_skill,
                strict_json_schema=True,
            ),
            FunctionTool(
                name=READ_SKILL_RESOURCE_TOOL_NAME,
                description=(
                    "Read one non-secret resource file contained inside an "
                    "already available local Agent Skill."
                ),
                params_json_schema=_read_skill_resource_schema(),
                on_invoke_tool=read_skill_resource,
                strict_json_schema=True,
            ),
        ]

    def _load_skill(
        self,
        profile: AgentProfile,
        run_id: int,
        arguments_json: str,
    ) -> dict[str, object]:
        arguments = _arguments(arguments_json)
        skill_name = _text_argument(arguments, "name")
        if denied := self._deny_if_unauthorized(
            profile=profile,
            run_id=run_id,
            tool_name=LOAD_SKILL_TOOL_NAME,
            arguments=arguments,
            skill_name=skill_name,
        ):
            return denied

        invocation = self._start_invocation(
            run_id,
            LOAD_SKILL_TOOL_NAME,
            arguments,
        )
        try:
            skill = self.service.load_skill(profile, skill_name)
        except Exception as error:
            return self._fail_invocation(invocation.id, error)

        self._complete_invocation(
            invocation=invocation,
            skill_name=skill.metadata.name.value,
            version=skill.metadata.version,
            content_hash=skill.metadata.content_hash,
            resource_name=None,
        )
        return _skill_response(skill)

    def _read_skill_resource(
        self,
        profile: AgentProfile,
        run_id: int,
        arguments_json: str,
    ) -> dict[str, object]:
        arguments = _arguments(arguments_json)
        skill_name = _text_argument(arguments, "skill_name")
        relative_path = _text_argument(arguments, "relative_path")
        if denied := self._deny_if_unauthorized(
            profile=profile,
            run_id=run_id,
            tool_name=READ_SKILL_RESOURCE_TOOL_NAME,
            arguments=arguments,
            skill_name=skill_name,
        ):
            return denied

        invocation = self._start_invocation(
            run_id,
            READ_SKILL_RESOURCE_TOOL_NAME,
            arguments,
        )
        try:
            content, content_hash = self.service.read_skill_resource(
                profile,
                skill_name,
                relative_path,
            )
        except Exception as error:
            return self._fail_invocation(invocation.id, error)

        self._complete_invocation(
            invocation=invocation,
            skill_name=skill_name,
            version=None,
            content_hash=content_hash,
            resource_name=relative_path,
        )
        return {
            "skill_name": skill_name,
            "resource_name": relative_path,
            "content_hash": content_hash,
            "content": content,
        }

    def _deny_if_unauthorized(
        self,
        profile: AgentProfile,
        run_id: int,
        tool_name: str,
        arguments: dict[str, object],
        skill_name: str,
    ) -> dict[str, object] | None:
        try:
            self.service.authorize_skill_access(profile, skill_name)
        except (AgentSkillAccessDeniedError, InvalidAgentSkillError) as error:
            arguments_hash, arguments_preview = sanitize_tool_arguments(
                tool_name,
                arguments,
            )
            error_type, error_message = sanitize_error(error)
            self.audit_repository.deny_tool_invocation(
                ToolInvocationDenial(
                    invocation=ToolInvocationStart(
                        run_id=run_id,
                        server_name=SKILL_SERVER_NAME,
                        tool_name=tool_name,
                        classification=ToolClassification.READ_ONLY,
                        arguments_hash=arguments_hash,
                        arguments_preview_json=arguments_preview,
                    ),
                    error_type=error_type,
                    error_message=error_message,
                ),
            )
            return {
                "error": (
                    f"{SKILL_ACCESS_DENIED_PREFIX}: {error_message}. "
                    "Do not retry with a different role or path."
                ),
            }
        return None

    def _start_invocation(
        self,
        run_id: int,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolInvocationRecord:
        arguments_hash, arguments_preview = sanitize_tool_arguments(
            tool_name,
            arguments,
        )
        return self.audit_repository.start_tool_invocation(
            ToolInvocationStart(
                run_id=run_id,
                server_name=SKILL_SERVER_NAME,
                tool_name=tool_name,
                classification=ToolClassification.READ_ONLY,
                arguments_hash=arguments_hash,
                arguments_preview_json=arguments_preview,
            ),
        )

    def _complete_invocation(
        self,
        invocation: ToolInvocationRecord,
        skill_name: str,
        version: str | None,
        content_hash: str,
        resource_name: str | None,
    ) -> None:
        result_payload = _skill_result_preview(
            name=skill_name,
            version=version,
            content_hash=content_hash,
            resource_name=resource_name,
        )
        self.audit_repository.complete_tool_invocation(
            invocation_id=invocation.id,
            result_hash=content_hash,
            result_preview=_json(result_payload),
        )

    def _fail_invocation(
        self,
        invocation_id: int,
        error: Exception,
    ) -> dict[str, object]:
        error_type, error_message = sanitize_error(error)
        self.audit_repository.fail_tool_invocation(
            invocation_id=invocation_id,
            error_type=error_type,
            error_message=error_message,
        )
        return {"error": f"SKILL_LOAD_FAILED: {error_message}"}


def _arguments(arguments_json: str) -> dict[str, object]:
    parsed = json.loads(arguments_json)
    if not isinstance(parsed, dict):
        raise InvalidAgentSkillError("Tool arguments must be a JSON object.")
    return dict(cast("dict[str, object]", parsed))


def _text_argument(arguments: dict[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidAgentSkillError(f"Tool argument {key} is required.")
    return value.strip()


def _load_skill_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of an available local Agent Skill.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    }


def _read_skill_resource_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of an available local Agent Skill.",
            },
            "relative_path": {
                "type": "string",
                "description": "Relative resource path inside the skill.",
            },
        },
        "required": ["skill_name", "relative_path"],
        "additionalProperties": False,
    }


def _skill_response(skill: AgentSkill) -> dict[str, object]:
    return {
        "name": skill.metadata.name.value,
        "description": skill.metadata.description,
        "version": skill.metadata.version,
        "content_hash": skill.metadata.content_hash,
        "instructions": skill.body,
    }


def _skill_result_preview(
    name: str,
    version: str | None,
    content_hash: str,
    resource_name: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "content_hash": content_hash,
        "loaded": True,
    }
    if version is not None:
        payload["version"] = version
    if resource_name is not None:
        payload["resource_name"] = resource_name
    return payload


def _json(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
