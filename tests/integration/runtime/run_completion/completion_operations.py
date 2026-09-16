"""Real workflow and restricted workspace operations for scripted segments."""

import json
from dataclasses import dataclass

from agent_team.application.audit.audit_sanitizer import hash_text
from agent_team.application.sessions.workspace_identity import (
    workspace_identity_hash,
)
from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.application.workspace.workspace_service import WorkspaceService
from agent_team.domain.audit.agent_audit_repository import AgentAuditRepository
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_denial import ToolInvocationDenial
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workspace.workspace_access_denied_error import (
    WorkspaceAccessDeniedError,
)


@dataclass(slots=True)
class CompletionOperations:
    """Execute authorized operations and record their sanitized audit facts."""

    workflow: WorkflowService
    workspace: WorkspaceService
    audit: AgentAuditRepository
    revision: int = 0
    discoveries: int = 0

    def perform(
        self,
        operation: str,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
    ) -> None:
        """Dispatch a test-selected operation using the harness binding."""
        assert task.task_id == run.task_id
        assert task.feature_id == run.feature_id
        assert task.role is profile.role is run.role
        {
            "activate": self._activate,
            "patch": self._patch,
            "check": self._check,
            "submit": self._submit,
            "discover": self._discover,
            "read_same": self._read_same,
            "denied_patch": self._denied_patch,
            "failed_patch": self._failed_patch,
            "block": self._block,
        }[operation](task, profile, run)

    def _activate(
        self, task: AgentTask, _profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        assert task.task_id is not None
        self.workflow.update_task_status(task.task_id, TaskStatus.IN_PROGRESS)
        self._record(
            run,
            "update_task_status",
            {"task_id": task.task_id, "status": "in_progress"},
            {"status": "in_progress"},
        )

    def _patch(
        self, task: AgentTask, profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        previous = f"return {self.revision}"
        following = f"return {self.revision + 1}"
        patch = self.workspace.apply_patch(
            profile, task, "backend/auth.py", previous, following
        )
        assert patch.applied
        self.revision += 1
        self._record(
            run,
            "apply_patch",
            {
                "path": patch.path,
                "old_text_hash": hash_text(previous),
                "new_text_hash": hash_text(following),
            },
            {
                "applied": patch.applied,
                "path": patch.path,
                "before_hash": patch.before_hash,
                "after_hash": patch.after_hash,
            },
        )

    def _check(
        self, task: AgentTask, profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        check = self.workspace.run_check(profile, task, "backend")
        assert check.exit_code == 0
        self._record(
            run,
            "run_check",
            {"name": check.name},
            {"name": check.name, "exit_code": check.exit_code},
        )

    def _submit(
        self, task: AgentTask, profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        assert task.task_id is not None
        assert task.workspace_root is not None
        handoff = self.workflow.submit_task_for_verification(
            TaskHandoffDraft(
                task_id=task.task_id,
                agent_run_id=run.id,
                submitted_by=profile.role,
                attribution=f"agent:{profile.role.value}",
                workspace_identity_hash=workspace_identity_hash(
                    task.workspace_root
                ),
                implementation_summary=f"Revision {self.revision}.",
                changed_paths=("backend/auth.py",),
                reused_symbols=("AuthService.logout",),
                new_symbols=(),
                reuse_notes="Extended the existing exact method.",
                checks_attempted=("backend",),
                limitations="none",
                next_action="deterministic verification",
            )
        )
        self._record(
            run,
            "submit_task_for_verification",
            {"task_id": task.task_id},
            {"submitted": True, "submission_id": handoff.id},
        )

    def _discover(
        self, task: AgentTask, profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        path = f"backend/discovery_{self.discoveries}.py"
        content = self.workspace.read_file(profile, task, path)
        self.discoveries += 1
        self._record(
            run,
            "read_file",
            {"path": path},
            {"path": path, "content_hash": content.content_hash},
        )

    def _read_same(
        self, task: AgentTask, profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        content = self.workspace.read_file(profile, task, "backend/auth.py")
        self._record(
            run,
            "read_file",
            {"path": content.path},
            {"path": content.path, "content_hash": content.content_hash},
        )

    def _denied_patch(
        self, task: AgentTask, profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        try:
            self.workspace.apply_patch(
                profile, task, "backend/auth.py", "return 0", "return 1"
            )
        except WorkspaceAccessDeniedError as error:
            self.audit.deny_tool_invocation(
                ToolInvocationDenial(
                    invocation=self._invocation(
                        run, "apply_patch", {"path": "backend/auth.py"}
                    ),
                    error_type=type(error).__name__,
                    error_message="Task activation is required.",
                )
            )
        else:
            raise AssertionError("The premature mutation was not denied.")

    def _failed_patch(
        self, task: AgentTask, profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        patch = self.workspace.apply_patch(
            profile, task, "backend/auth.py", "absent text", "return 1"
        )
        assert not patch.applied
        self._record(
            run,
            "apply_patch",
            {"path": patch.path, "old_text_hash": hash_text("absent text")},
            {
                "applied": False,
                "path": patch.path,
                "before_hash": patch.before_hash,
                "after_hash": patch.after_hash,
            },
        )

    def _block(
        self, task: AgentTask, _profile: AgentProfile, run: AgentRunRecord
    ) -> None:
        assert task.task_id is not None
        self.workflow.update_task_status(task.task_id, TaskStatus.BLOCKED)
        self._record(
            run,
            "update_task_status",
            {"task_id": task.task_id, "status": "blocked"},
            {"status": "blocked"},
        )

    def _record(
        self,
        run: AgentRunRecord,
        name: str,
        arguments: dict[str, object],
        result: dict[str, object],
    ) -> None:
        invocation = self.audit.start_tool_invocation(
            self._invocation(run, name, arguments)
        )
        encoded_result = json.dumps(result, sort_keys=True)
        self.audit.complete_tool_invocation(
            invocation_id=invocation.id,
            result_hash=hash_text(encoded_result),
            result_preview=encoded_result,
        )

    def _invocation(
        self, run: AgentRunRecord, name: str, arguments: dict[str, object]
    ) -> ToolInvocationStart:
        encoded_arguments = json.dumps(arguments, sort_keys=True)
        workflow_tool = name in {
            "update_task_status",
            "submit_task_for_verification",
        }
        return ToolInvocationStart(
            run_id=run.id,
            server_name="development_workflow"
            if workflow_tool
            else "workspace",
            tool_name=name,
            classification=(
                ToolClassification.READ_ONLY
                if name in {"read_file", "run_check"}
                else ToolClassification.MUTATING
            ),
            arguments_hash=hash_text(encoded_arguments),
            arguments_preview_json=encoded_arguments,
        )
