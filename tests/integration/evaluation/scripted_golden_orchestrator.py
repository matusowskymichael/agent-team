"""Offline developer actions for exercising real evaluation persistence."""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass

from agent_team.application.runtime.agent_profile_catalog import (
    AgentProfileCatalog,
)
from agent_team.application.sessions.workspace_identity import (
    workspace_identity_hash,
)
from agent_team.application.workflow.task_verification_service import (
    TaskVerificationService,
)
from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.application.workspace.workspace_service import WorkspaceService
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.evaluation.evaluation_task_verifier import (
    EvaluationTaskVerifier,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)

AuditRepository = audit_repository_module.SQLiteAgentAuditRepository
WorkflowRepository = workflow_repository_module.SQLiteWorkflowRepository


@dataclass(frozen=True)
class ScriptedGoldenOrchestrator:
    """Perform real workspace/workflow operations without model inference."""

    repository: WorkflowRepository
    audit: AuditRepository
    case: EvalCase
    implementation: str

    async def run(self, task: AgentTask) -> AgentResult:
        """Exercise a developer edit and optional verified submission."""
        assert task.workspace_root is not None
        run = self.audit.start_run(
            AgentRunStart(
                role=task.role,
                model="qwen3.5:9b",
                prompt_hash="scripted",
                prompt_excerpt="Offline golden regression.",
                max_turns=10,
                feature_id=task.feature_id,
                task_id=task.task_id,
                workspace_identity_hash=workspace_identity_hash(
                    task.workspace_root,
                ),
            ),
        )
        path, check_name = self._edit(task, run.id)
        if any(
            call.name == "submit_task_for_verification"
            for call in self.case.expected_tool_calls
        ):
            self._submit(task, run.id, path, check_name)
        return AgentResult(
            response="; ".join(self.case.objective_response_facts)
        )

    def _edit(self, task: AgentTask, run_id: int) -> tuple[str, str]:
        assert task.workspace_root is not None
        assert task.task_id is not None
        task_id = task.task_id
        profile = AgentProfileCatalog().get_profile(task.role)
        check_name = task.role.value.split("_")[0]
        workspace = WorkspaceService(
            repository=self.repository,
            executor=LocalWorkspaceExecutor(
                root=task.workspace_root,
                check_commands={
                    check_name: (
                        sys.executable,
                        "-c",
                        "print('fixture check')",
                    ),
                },
            ),
        )
        path = _argument(self.case, "apply_patch", "path")
        symbol = _argument(self.case, "find_symbol", "name")
        self._observe(
            run_id,
            "list_files",
            {},
            lambda: workspace.list_files(
                profile,
                task,
            ),
        )
        self._observe(
            run_id,
            "search_code",
            {"query": symbol},
            lambda: workspace.search_code(profile, task, symbol),
        )
        self._observe(
            run_id,
            "find_symbol",
            {"name": symbol},
            lambda: workspace.find_symbol(profile, task, symbol),
        )
        before = ""
        if (task.workspace_root / path).is_file():
            before = workspace.read_file(profile, task, path).content
            self._observe(
                run_id,
                "read_file",
                {"path": path},
                lambda: workspace.read_file(profile, task, path),
            )
        workflow = WorkflowService(self.repository)
        self._observe(
            run_id,
            "update_task_status",
            {"task_id": task.task_id, "status": "in_progress"},
            lambda: workflow.update_task_status(
                task_id,
                TaskStatus.IN_PROGRESS,
            ),
        )
        self._observe(
            run_id,
            "apply_patch",
            {"path": path},
            lambda: workspace.apply_patch(
                profile,
                task,
                path,
                before,
                self.implementation,
            ),
        )
        self._observe(
            run_id,
            "run_check",
            {"name": check_name},
            lambda: workspace.run_check(profile, task, check_name),
        )
        return path, check_name

    def _submit(
        self,
        task: AgentTask,
        run_id: int,
        path: str,
        check_name: str,
    ) -> None:
        assert task.task_id is not None
        assert task.workspace_root is not None
        draft = TaskHandoffDraft(
            task_id=task.task_id,
            agent_run_id=run_id,
            submitted_by=task.role,
            attribution=f"agent:{task.role.value}",
            workspace_identity_hash=workspace_identity_hash(
                task.workspace_root
            ),
            implementation_summary="Implemented assigned behavior.",
            changed_paths=(path,),
            reused_symbols=(),
            new_symbols=(),
            reuse_notes="Inspected existing code before applying the patch.",
            checks_attempted=(check_name,),
            limitations="none",
            next_action="Verify the submitted workspace.",
        )
        self._observe(
            run_id,
            "submit_task_for_verification",
            {"task_id": task.task_id},
            lambda: WorkflowService(
                self.repository
            ).submit_task_for_verification(draft),
        )
        TaskVerificationService(
            repository=self.repository,
            verifier=EvaluationTaskVerifier(self.case),
            audit_reader=self.audit,
        ).verify_task(task.task_id, task.workspace_root)

    def _observe(
        self,
        run_id: int,
        name: str,
        arguments: dict[str, object],
        operation: Callable[[], object],
    ) -> None:
        invocation = self.audit.start_tool_invocation(
            ToolInvocationStart(
                run_id=run_id,
                server_name="development_workflow",
                tool_name=name,
                classification=ToolClassification.MUTATING,
                arguments_hash="scripted",
                arguments_preview_json=json.dumps(arguments),
            ),
        )
        operation()
        self.audit.complete_tool_invocation(
            invocation_id=invocation.id,
            result_hash="scripted",
            result_preview="{}",
        )


def _argument(case: EvalCase, tool_name: str, argument_name: str) -> str:
    call = next(
        call for call in case.expected_tool_calls if call.name == tool_name
    )
    value = call.arguments_subset[argument_name]
    assert isinstance(value, str)
    return value
