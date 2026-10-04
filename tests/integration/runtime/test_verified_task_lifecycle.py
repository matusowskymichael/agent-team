"""Integration tests for verified task lifecycle orchestration."""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_team.application.audit.audit_sanitizer import hash_text
from agent_team.application.context.feature_context_builder import (
    FeatureContextBuilder,
)
from agent_team.application.runtime.agent_harness import AgentHarness
from agent_team.application.sessions.agent_session_service import (
    AgentSessionService,
)
from agent_team.application.sessions.workspace_identity import (
    workspace_identity_hash,
)
from agent_team.application.workflow.task_verification_service import (
    TaskVerificationService,
)
from agent_team.application.workflow.workflow_service import WorkflowService
from agent_team.application.workspace.workspace_service import (
    WorkspaceService,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.context.agent_context_envelope import (
    AgentContextEnvelope,
)
from agent_team.domain.runtime.agent_profile import AgentProfile
from agent_team.domain.runtime.agent_result import AgentResult
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.runtime.agent_task import AgentTask
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.runtime.workflow_tool_name import WorkflowToolName
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_profiles import (
    BACKEND_REQUIRED_CHECKS,
)
from agent_team.domain.workspace.workspace_tool_name import WorkspaceToolName
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.sessions import (
    sqlite_agent_session_repository as session_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_repository as workflow_repository_module,
)
from agent_team.infrastructure.workspace.local_task_verifier import (
    LocalTaskVerifier,
)
from agent_team.infrastructure.workspace.local_workspace_executor import (
    LocalWorkspaceExecutor,
)

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)
AuditRepository = audit_repository_module.SQLiteAgentAuditRepository
SessionRepository = session_repository_module.SQLiteAgentSessionRepository
WorkflowRepository = workflow_repository_module.SQLiteWorkflowRepository


@dataclass(frozen=True, slots=True)
class _RecordedTool:
    tool_name: str
    arguments: dict[str, object]
    result: dict[str, object]
    classification: ToolClassification = ToolClassification.MUTATING
    server_name: str = "development_workflow"


@dataclass(slots=True)
class _LifecycleRuntime:
    workflow: WorkflowService
    workflow_repository: WorkflowRepository
    audit_repository: AuditRepository
    check_commands: dict[str, tuple[str, ...]]
    fail_after_submit: bool = False

    @property
    def model_name(self) -> str:
        """Return the fake local model name."""
        return "fake-model"

    async def execute(
        self,
        task: AgentTask,
        profile: AgentProfile,
        run: AgentRunRecord,
        context: AgentContextEnvelope | None = None,
        skill_context: str | None = None,
    ) -> AgentResult:
        """Perform the developer's trusted lifecycle tool sequence."""
        assert task.task_id is not None
        assert task.workspace_root is not None
        assert context is None or context.task_id == task.task_id
        assert skill_context is None
        if task.continuation_context is not None:
            return AgentResult(response="No repair was attempted.")
        self.workflow.update_task_status(task.task_id, TaskStatus.IN_PROGRESS)
        _record_tool(
            self.audit_repository,
            run.id,
            _RecordedTool(
                tool_name=WorkflowToolName.UPDATE_TASK_STATUS.value,
                arguments={"task_id": task.task_id, "status": "in_progress"},
                result={"status": "in_progress"},
            ),
        )
        workspace = WorkspaceService(
            repository=self.workflow_repository,
            executor=LocalWorkspaceExecutor(
                root=task.workspace_root,
                check_commands=self.check_commands,
            ),
        )
        patch = workspace.apply_patch(
            profile,
            task,
            "backend/auth.py",
            "return bool(token)",
            "return bool(token) and token != 'revoked'",
        )
        _record_tool(
            self.audit_repository,
            run.id,
            _RecordedTool(
                tool_name=WorkspaceToolName.APPLY_PATCH.value,
                arguments={"path": patch.path},
                result={"applied": patch.applied, "path": patch.path},
                server_name="workspace",
            ),
        )
        check = workspace.run_check(profile, task, "backend")
        _record_tool(
            self.audit_repository,
            run.id,
            _RecordedTool(
                tool_name=WorkspaceToolName.RUN_CHECK.value,
                arguments={"name": check.name},
                result={"exit_code": check.exit_code},
                classification=ToolClassification.READ_ONLY,
                server_name="workspace",
            ),
        )
        self.workflow.submit_task_for_verification(
            TaskHandoffDraft(
                task_id=task.task_id,
                agent_run_id=run.id,
                submitted_by=profile.role,
                attribution=f"agent:{profile.role.value}",
                workspace_identity_hash=workspace_identity_hash(
                    task.workspace_root,
                ),
                implementation_summary="Extended AuthService.logout.",
                changed_paths=(patch.path,),
                reused_symbols=("AuthService.logout",),
                new_symbols=(),
                reuse_notes="Existing logout method was extended.",
                checks_attempted=(check.name,),
                limitations="none",
                next_action="run deterministic verification",
            ),
        )
        _record_tool(
            self.audit_repository,
            run.id,
            _RecordedTool(
                tool_name=WorkflowToolName.SUBMIT_TASK_FOR_VERIFICATION.value,
                arguments={"task_id": task.task_id},
                result={"submitted": True},
            ),
        )
        if self.fail_after_submit:
            raise RuntimeError("runtime failed after persisted submission")
        return AgentResult(response="Submitted for verification.")


class TestVerifiedTaskLifecycleIntegration:
    """End-to-end lifecycle behavior with real SQLite persistence."""

    def test_successful_submission_completes_after_verification(
        self,
        tmp_path: Path,
    ) -> None:
        """Complete only after deterministic verification passes."""
        database_path = tmp_path / "workflow.db"
        workspace_root = _workspace(tmp_path)
        workflow_repository, workflow, task_id = _seed_task(database_path)
        audit_repository = AuditRepository(database_path)
        harness = _harness(
            workflow_repository=workflow_repository,
            workflow=workflow,
            audit_repository=audit_repository,
            check_commands=_passing_checks(),
            verification_commands=_passing_checks(),
        )

        result = asyncio.run(
            harness.execute(_agent_task(task_id, workspace_root)),
        )

        task = workflow_repository.get_task(task_id)
        handoff = workflow_repository.latest_task_handoff(task_id)
        evidence = workflow_repository.latest_task_verification(task_id)
        run = audit_repository.list_runs(limit=1)[0]

        assert task is not None
        assert task.status is TaskStatus.COMPLETED
        assert handoff is not None
        assert handoff.changed_paths == ("backend/auth.py",)
        assert evidence is not None
        assert evidence.outcome is TaskVerificationOutcome.PASSED
        assert tuple(check.name for check in evidence.checks) == (
            BACKEND_REQUIRED_CHECKS
        )
        assert "Verification result: passed" in result.response
        assert run.task_id == task_id
        assert run.workspace_identity_hash is not None
        assert str(workspace_root) not in run.workspace_identity_hash

    def test_failed_verification_returns_task_to_in_progress(
        self,
        tmp_path: Path,
    ) -> None:
        """Reopen submitted work when independent verification fails."""
        database_path = tmp_path / "workflow.db"
        workspace_root = _workspace(tmp_path)
        workflow_repository, workflow, task_id = _seed_task(database_path)
        harness = _harness(
            workflow_repository=workflow_repository,
            workflow=workflow,
            audit_repository=AuditRepository(database_path),
            check_commands=_passing_checks(),
            verification_commands=_failing_checks(),
        )

        with pytest.raises(AgentStalledError):
            asyncio.run(
                harness.execute(_agent_task(task_id, workspace_root)),
            )

        task = workflow_repository.get_task(task_id)
        evidence = workflow_repository.latest_task_verification(task_id)

        assert task is not None
        assert task.status is TaskStatus.IN_PROGRESS
        assert evidence is not None
        assert evidence.outcome is TaskVerificationOutcome.FAILED
        assert evidence.failure_classification is (
            FailureClassification.CHECK_FAILED
        )

    def test_interrupted_submission_resumes_verification(
        self,
        tmp_path: Path,
    ) -> None:
        """Resume a persisted verification_pending handoff idempotently."""
        database_path = tmp_path / "workflow.db"
        workspace_root = _workspace(tmp_path)
        workflow_repository, workflow, task_id = _seed_task(database_path)
        audit_repository = AuditRepository(database_path)
        runtime = _LifecycleRuntime(
            workflow=workflow,
            workflow_repository=workflow_repository,
            audit_repository=audit_repository,
            check_commands=_passing_checks(),
            fail_after_submit=True,
        )
        harness = AgentHarness(
            runtime=runtime,
            audit_repository=audit_repository,
            session_service=AgentSessionService(
                repository=SessionRepository(database_path),
                workflow_repository=workflow_repository,
            ),
            context_provider=FeatureContextBuilder(workflow_repository),
        )

        with pytest.raises(RuntimeError, match="persisted submission"):
            asyncio.run(harness.execute(_agent_task(task_id, workspace_root)))

        task = workflow_repository.get_task(task_id)
        assert task is not None
        assert task.status is TaskStatus.VERIFICATION_PENDING

        verifier = TaskVerificationService(
            repository=workflow_repository,
            verifier=LocalTaskVerifier(
                executor_factory=lambda root: LocalWorkspaceExecutor(
                    root=root,
                    check_commands=_passing_checks(),
                ),
            ),
            audit_reader=audit_repository,
        )
        first_evidence = verifier.verify_task(task_id, workspace_root)
        second_evidence = verifier.verify_task(task_id, workspace_root)
        completed_task = workflow_repository.get_task(task_id)

        assert completed_task is not None
        assert completed_task.status is TaskStatus.COMPLETED
        assert first_evidence == second_evidence
        assert first_evidence.outcome is TaskVerificationOutcome.PASSED

    @pytest.mark.parametrize("stream", ("stdout", "stderr"))
    def test_invalid_check_output_blocks_verification(
        self,
        tmp_path: Path,
        stream: str,
    ) -> None:
        """Persist blocked evidence when a real check emits invalid bytes."""
        database_path = tmp_path / "workflow.db"
        workspace_root = _workspace(tmp_path)
        workflow_repository, workflow, task_id = _seed_task(database_path)
        verification_commands = _passing_checks()
        verification_commands[BACKEND_REQUIRED_CHECKS[0]] = (
            sys.executable,
            "-c",
            "print('token=private-value')",
        )
        verification_commands[BACKEND_REQUIRED_CHECKS[1]] = (
            sys.executable,
            "-c",
            f"import sys; sys.{stream}.buffer.write("
            "b'token=invalid-output-value\\xff')",
        )
        harness = _harness(
            workflow_repository=workflow_repository,
            workflow=workflow,
            audit_repository=AuditRepository(database_path),
            check_commands=_passing_checks(),
            verification_commands=verification_commands,
        )

        result = asyncio.run(
            harness.execute(_agent_task(task_id, workspace_root)),
        )
        reopened_repository = WorkflowRepository(database_path)
        task = reopened_repository.get_task(task_id)
        evidence = reopened_repository.latest_task_verification(task_id)

        assert task is not None
        assert task.status is TaskStatus.BLOCKED
        assert evidence is not None
        assert evidence.outcome is TaskVerificationOutcome.BLOCKED
        assert evidence.failure_classification is (
            FailureClassification.INFRASTRUCTURE_ERROR
        )
        assert "UnicodeDecodeError" in evidence.feedback
        assert tuple(check.name for check in evidence.checks) == (
            BACKEND_REQUIRED_CHECKS[0],
        )
        assert evidence.checks[0].stdout_excerpt == "token=[REDACTED]"
        assert evidence.checks[0].stdout_hash == hash_text("token=[REDACTED]")
        assert "private-value" not in repr(evidence)
        assert "invalid-output-value" not in repr(evidence)
        assert "invalid-output-value" not in result.response
        assert "Verification result: blocked" in result.response


def _harness(
    workflow_repository: WorkflowRepository,
    workflow: WorkflowService,
    audit_repository: AuditRepository,
    check_commands: dict[str, tuple[str, ...]],
    verification_commands: dict[str, tuple[str, ...]],
) -> AgentHarness:
    return AgentHarness(
        runtime=_LifecycleRuntime(
            workflow=workflow,
            workflow_repository=workflow_repository,
            audit_repository=audit_repository,
            check_commands=check_commands,
        ),
        audit_repository=audit_repository,
        session_service=AgentSessionService(
            repository=SessionRepository(workflow_repository.database_path),
            workflow_repository=workflow_repository,
        ),
        context_provider=FeatureContextBuilder(workflow_repository),
        task_verification_service=TaskVerificationService(
            repository=workflow_repository,
            verifier=LocalTaskVerifier(
                executor_factory=lambda root: LocalWorkspaceExecutor(
                    root=root,
                    check_commands=verification_commands,
                ),
            ),
            audit_reader=audit_repository,
        ),
    )


def _seed_task(
    database_path: Path,
) -> tuple[WorkflowRepository, WorkflowService, int]:
    repository = WorkflowRepository(database_path)
    workflow = WorkflowService(repository)
    feature = workflow.create_feature("Logout", "Invalidate sessions.")
    workflow.add_artifact(
        feature_id=feature.id,
        kind=ArtifactKind.REQUIREMENTS,
        content="Logout must revoke the active token.",
        created_by="agent:business_analyst",
    )
    task = workflow.create_task(
        feature.id,
        "Add token revocation",
        "Extend AuthService.logout.",
        DevelopmentRole.BACKEND_DEVELOPER,
    )
    return repository, workflow, task.id


def _workspace(tmp_path: Path) -> Path:
    workspace_root = tmp_path / "workspace"
    backend_file = workspace_root / "backend" / "auth.py"
    backend_file.parent.mkdir(parents=True, exist_ok=True)
    backend_file.write_text(
        (
            "class AuthService:\n"
            "    def logout(self, token: str) -> bool:\n"
            "        return bool(token)\n"
        ),
        encoding="utf-8",
    )
    return workspace_root


def _agent_task(task_id: int, workspace_root: Path) -> AgentTask:
    return AgentTask(
        prompt="Implement the assigned backend task.",
        role=DevelopmentRole.BACKEND_DEVELOPER,
        feature_id=1,
        task_id=task_id,
        workspace_root=workspace_root,
    )


def _passing_checks() -> dict[str, tuple[str, ...]]:
    pass_command = (sys.executable, "-c", "pass")
    return {
        "backend": pass_command,
        **dict.fromkeys(BACKEND_REQUIRED_CHECKS, pass_command),
    }


def _failing_checks() -> dict[str, tuple[str, ...]]:
    pass_command = (sys.executable, "-c", "pass")
    fail_command = (sys.executable, "-c", "raise SystemExit(1)")
    return {
        "backend": pass_command,
        **dict.fromkeys(BACKEND_REQUIRED_CHECKS, fail_command),
    }


def _record_tool(
    audit_repository: AuditRepository,
    run_id: int,
    tool: _RecordedTool,
) -> None:
    invocation = audit_repository.start_tool_invocation(
        ToolInvocationStart(
            run_id=run_id,
            server_name=tool.server_name,
            tool_name=tool.tool_name,
            classification=tool.classification,
            arguments_hash="arguments-hash",
            arguments_preview_json=json.dumps(
                tool.arguments,
                separators=(",", ":"),
            ),
        ),
    )
    audit_repository.complete_tool_invocation(
        invocation_id=invocation.id,
        result_hash="result-hash",
        result_preview=json.dumps(tool.result, separators=(",", ":")),
    )
