"""Offline side effects around cancellation for candidate evidence tests."""

import asyncio
from dataclasses import dataclass

from agent_team.application.audit.audit_sanitizer import (
    sanitize_tool_arguments,
    sanitize_tool_result,
)
from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.runtime.agent_cleanup_timeout_error import (
    AgentCleanupTimeoutError,
)
from agent_team.domain.runtime.agent_provider_timeout_error import (
    AgentProviderTimeoutError,
)
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_segment_timeout_error import (
    AgentSegmentTimeoutError,
)
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.mcp.client import (
    workflow_mcp_unavailable_error,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)


@dataclass
class CancelAfterAuditOrchestrator:
    """Perform deterministic trusted effects, then stop at a selected point."""

    workflow: workflow_repository_module.SQLiteWorkflowRepository
    audit: audit_repository_module.SQLiteAgentAuditRepository
    boundary: str
    tasks: list[AgentTask]

    async def run(self, task: AgentTask) -> AgentResult:
        """Model uncertainty before or after persisted patches and handoffs."""
        self.tasks.append(task)
        if self.boundary == "completed":
            return AgentResult("Advisory response.")
        assert task.task_id is not None
        assert task.workspace_root is not None
        run = self.audit.start_run(
            AgentRunStart(
                role=task.role,
                model="qwen3.5:9b",
                prompt_hash="safe-hash",
                prompt_excerpt="",
                max_turns=10,
                feature_id=task.feature_id,
                task_id=task.task_id,
                workspace_identity_hash="trusted-workspace",
            )
        )
        self.audit.record_run_progress(run.id, 2)
        WorkflowService(self.workflow).update_task_status(
            task.task_id,
            TaskStatus.IN_PROGRESS,
        )
        self._record_operation(
            run.id,
            "update_task_status",
            {
                "task_id": task.task_id,
                "status": "in_progress",
            },
            {"status": "in_progress"},
        )
        patch_result: dict[str, object] | None = None
        if self.boundary != "before_patch":
            target = task.workspace_root / "backend/auth_service.py"
            target.write_text("class AuthService: pass\n", encoding="utf-8")
            patch_result = {
                "path": "backend/auth_service.py",
                "applied": True,
                "before_hash": "old-safe-hash",
                "after_hash": "new-safe-hash",
            }
        self._record_operation(
            run.id,
            "apply_patch",
            {
                "path": "backend/auth_service.py",
                "old_text": "fixture-private-source-before",
                "new_text": (
                    "fixture-private-source-after api_key=private-value"
                ),
                "diagnostics": [
                    {
                        "prompt": "fixture-private-prompt",
                        "nested": [
                            {
                                "path": str(
                                    task.workspace_root / "private.py"
                                ),
                                "content": "fixture-private-content",
                            }
                        ],
                    }
                ],
            },
            patch_result,
        )
        if self.boundary in {"before_handoff", "after_handoff"}:
            self._submit_handoff(task, run.id)
        if self.boundary == "unsafe_metadata":
            self._record_operation(
                run.id,
                "apply_patch",
                {"path": r"C:\private\file.py"},
                {"path": r"C:\private\file.py", "applied": True},
            )
            self._record_operation(
                run.id, "run_check", {"name": "backend"}, {"name": "backend"}
            )
            self._record_operation(
                run.id, "private-tool /private/workspace", {}, {}
            )
        if self.boundary == "provider_timeout":
            raise AgentProviderTimeoutError("Local provider deadline reached.")
        if self.boundary == "segment_timeout":
            raise AgentSegmentTimeoutError("Local segment deadline reached.")
        if self.boundary == "cleanup_timeout":
            raise AgentCleanupTimeoutError("Owned cleanup deadline reached.")
        if self.boundary in {
            "runtime_failure",
            "stalled",
            "infrastructure_failure",
        }:
            WorkflowService(self.workflow).create_feature(
                title=(
                    "Sensitive /private/workspace api_key=private-title-key "
                    + "long title " * 30
                ),
                description="Private source body.",
            )
            errors = {
                "runtime_failure": RuntimeError,
                "stalled": AgentStalledError,
                "infrastructure_failure": (
                    workflow_mcp_unavailable_error.WorkflowMCPUnavailableError
                ),
            }
            raise errors[self.boundary](
                "Stopped at /private/workspace api_key=private-error-key"
            )
        raise asyncio.CancelledError

    def _submit_handoff(self, task: AgentTask, run_id: int) -> None:
        assert task.task_id is not None
        self._record_operation(
            run_id,
            "run_check",
            {"name": "backend"},
            {
                "name": "backend",
                "exit_code": 0,
                "timed_out": False,
                "stdout_excerpt": "fixture-private-check-output",
                "stderr_excerpt": "",
            },
        )
        result: dict[str, object] | None = None
        if self.boundary == "after_handoff":
            handoff = WorkflowService(
                self.workflow
            ).submit_task_for_verification(
                TaskHandoffDraft(
                    task_id=task.task_id,
                    agent_run_id=run_id,
                    submitted_by=task.role,
                    attribution=f"agent:{task.role.value}",
                    workspace_identity_hash="trusted-workspace",
                    implementation_summary="Adjusted backend behavior.",
                    changed_paths=("backend/auth_service.py",),
                    reused_symbols=("AuthService.logout",),
                    new_symbols=(),
                    reuse_notes="Extended the existing service.",
                    checks_attempted=("backend",),
                    limitations="none",
                    next_action="verify",
                ),
            )
            result = {"id": handoff.id, "task_id": task.task_id}
        self._record_operation(
            run_id,
            "submit_task_for_verification",
            {
                "task_id": task.task_id,
            },
            result,
        )

    def _record_operation(
        self,
        run_id: int,
        name: str,
        arguments: dict[str, object],
        result: dict[str, object] | None,
    ) -> None:
        arguments_hash, preview = sanitize_tool_arguments(name, arguments)
        invocation = self.audit.start_tool_invocation(
            ToolInvocationStart(
                run_id=run_id,
                server_name="workspace",
                tool_name=name,
                classification=ToolClassification.MUTATING,
                arguments_hash=arguments_hash,
                arguments_preview_json=preview,
            )
        )
        if result is not None:
            result_hash, result_preview = sanitize_tool_result(name, result)
            self.audit.complete_tool_invocation(
                invocation.id,
                result_hash,
                result_preview,
            )
