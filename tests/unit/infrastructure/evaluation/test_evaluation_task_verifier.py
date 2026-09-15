"""Tests for deterministic evaluation task verification."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_feature_fixture import (
    EvalFeatureFixture,
)
from agent_team.domain.evaluation.eval_task_fixture import EvalTaskFixture
from agent_team.domain.evaluation.eval_workspace_file_fixture import (
    EvalWorkspaceFileFixture,
)
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)
from agent_team.infrastructure.evaluation.evaluation_task_verifier import (
    EvaluationTaskVerifier,
)


class TestEvaluationTaskVerifier:
    """Deterministic evaluation verifier behavior tests."""

    def test_backend_correct_change_passes(
        self,
        tmp_path: Path,
    ) -> None:
        """Accept a backend implementation with expected reuse evidence."""
        case = _case(
            role=DevelopmentRole.BACKEND_DEVELOPER,
            path="backend/auth.py",
            baseline="class AuthService:\n    pass\n",
            user_input="Revoke refresh tokens in the auth service.",
            task_description="Reuse AuthService.revoke_token.",
            symbol="AuthService.revoke_token",
        )
        _write(
            tmp_path,
            "backend/auth.py",
            "class AuthService:\n"
            "    def revoke_token(self) -> None:\n"
            "        self.revoked = True\n",
        )

        result = EvaluationTaskVerifier(case).verify(
            _task(DevelopmentRole.BACKEND_DEVELOPER),
            _handoff(
                DevelopmentRole.BACKEND_DEVELOPER,
                ("backend/auth.py",),
            ),
            tmp_path,
        )

        assert result.outcome is TaskVerificationOutcome.PASSED
        assert [check.name for check in result.checks] == [
            "backend",
            "backend:workspace-diff",
            "backend:symbols",
            "backend:behavior-markers",
        ]

    def test_backend_no_op_change_fails(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject a submitted backend task with unchanged workspace files."""
        case = _case(
            role=DevelopmentRole.BACKEND_DEVELOPER,
            path="backend/auth.py",
            baseline="def revoke_token() -> None:\n    pass\n",
            user_input="Revoke refresh tokens.",
            task_description="Patch the revoke behavior.",
            symbol=None,
        )
        _write(tmp_path, "backend/auth.py", case.workspace_files[0].content)

        result = EvaluationTaskVerifier(case).verify(
            _task(DevelopmentRole.BACKEND_DEVELOPER),
            _handoff(
                DevelopmentRole.BACKEND_DEVELOPER,
                ("backend/auth.py",),
            ),
            tmp_path,
        )

        assert result.outcome is TaskVerificationOutcome.FAILED
        assert _failed_check_names(result) == ["backend:workspace-diff"]

    def test_backend_wrong_symbol_change_fails(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject plausible backend edits that omit the expected symbol."""
        case = _case(
            role=DevelopmentRole.BACKEND_DEVELOPER,
            path="backend/auth.py",
            baseline="class AuthService:\n    pass\n",
            user_input="Revoke refresh tokens in the auth service.",
            task_description="Reuse AuthService.revoke_token.",
            symbol="AuthService.revoke_token",
        )
        _write(
            tmp_path,
            "backend/auth.py",
            "class AuthService:\n"
            "    def logout(self) -> None:\n"
            "        self.revoked = True\n",
        )

        result = EvaluationTaskVerifier(case).verify(
            _task(DevelopmentRole.BACKEND_DEVELOPER),
            _handoff(
                DevelopmentRole.BACKEND_DEVELOPER,
                ("backend/auth.py",),
            ),
            tmp_path,
        )

        assert result.outcome is TaskVerificationOutcome.FAILED
        assert _failed_check_names(result) == ["backend:symbols"]

    def test_frontend_correct_change_passes(
        self,
        tmp_path: Path,
    ) -> None:
        """Accept a frontend implementation with expected behavior marker."""
        case = _case(
            role=DevelopmentRole.FRONTEND_DEVELOPER,
            path="frontend/AccountMenu.tsx",
            baseline=(
                "export function AccountMenu() {\n"
                "  return <button>Account</button>\n"
                "}\n"
            ),
            user_input="Wire AccountMenu to call onLogout.",
            task_description="Extend AccountMenu with onLogout.",
            symbol="AccountMenu",
        )
        _write(
            tmp_path,
            "frontend/AccountMenu.tsx",
            "export function AccountMenu({ onLogout }) {\n"
            "  return <button onClick={onLogout}>Logout</button>\n"
            "}\n",
        )

        result = EvaluationTaskVerifier(case).verify(
            _task(DevelopmentRole.FRONTEND_DEVELOPER),
            _handoff(
                DevelopmentRole.FRONTEND_DEVELOPER,
                ("frontend/AccountMenu.tsx",),
            ),
            tmp_path,
        )

        assert result.outcome is TaskVerificationOutcome.PASSED
        assert [check.name for check in result.checks] == [
            "frontend",
            "frontend:workspace-diff",
            "frontend:symbols",
            "frontend:behavior-markers",
        ]

    def test_frontend_missing_behavior_marker_fails(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject frontend edits that miss the requested behavior."""
        case = _case(
            role=DevelopmentRole.FRONTEND_DEVELOPER,
            path="frontend/AccountMenu.tsx",
            baseline="export function AccountMenu() { return null }\n",
            user_input="Wire AccountMenu to call onLogout.",
            task_description="Extend AccountMenu with onLogout.",
            symbol="AccountMenu",
        )
        _write(
            tmp_path,
            "frontend/AccountMenu.tsx",
            "export function AccountMenu() {\n"
            "  return <button>Logout</button>\n"
            "}\n",
        )

        result = EvaluationTaskVerifier(case).verify(
            _task(DevelopmentRole.FRONTEND_DEVELOPER),
            _handoff(
                DevelopmentRole.FRONTEND_DEVELOPER,
                ("frontend/AccountMenu.tsx",),
            ),
            tmp_path,
        )

        assert result.outcome is TaskVerificationOutcome.FAILED
        assert _failed_check_names(result) == [
            "frontend:behavior-markers",
        ]

    def test_missing_expected_patch_assertions_block_verification(
        self,
        tmp_path: Path,
    ) -> None:
        """Fail closed when an eval case has no patch assertions."""
        case = replace(
            _case(
                role=DevelopmentRole.BACKEND_DEVELOPER,
                path="backend/auth.py",
                baseline="",
                user_input="Patch backend behavior.",
                task_description="Patch backend behavior.",
                symbol=None,
            ),
            expected_tool_calls=(),
        )

        result = EvaluationTaskVerifier(case).verify(
            _task(DevelopmentRole.BACKEND_DEVELOPER),
            _handoff(DevelopmentRole.BACKEND_DEVELOPER, ()),
            tmp_path,
        )

        assert result.outcome is TaskVerificationOutcome.BLOCKED
        assert result.checks == ()


def _case(  # noqa: PLR0913, PLR0917
    role: DevelopmentRole,
    path: str,
    baseline: str,
    user_input: str,
    task_description: str,
    symbol: str | None,
) -> EvalCase:
    expected_calls = [ExpectedToolCall("apply_patch", {"path": path})]
    if symbol is not None:
        expected_calls.append(
            ExpectedToolCall("find_symbol", {"name": symbol}),
        )
    return EvalCase(
        id="developer-eval",
        name="Developer eval",
        category="implementation",
        severity="critical",
        active_role=role,
        feature_fixtures=(
            EvalFeatureFixture(
                id=1,
                title="Feature",
                description="Description.",
                status=FeatureStatus.IMPLEMENTATION,
                artifacts=(),
                tasks=(
                    EvalTaskFixture(
                        feature_id=1,
                        title="Task",
                        description=task_description,
                        assigned_role=role,
                        status=TaskStatus.PENDING,
                    ),
                ),
            ),
        ),
        prior_session_turns=(),
        user_input=user_input,
        expected_tool_calls=tuple(expected_calls),
        forbidden_tool_calls=(),
        expected_database_effects=(),
        forbidden_database_effects=(),
        required_response_facts=(),
        forbidden_response_claims=(),
        rubric_id=f"{role.value}_workflow",
        note="Verifier unit test.",
        task_scope_id=1,
        feature_scope_id=1,
        workspace_files=(
            EvalWorkspaceFileFixture(path=path, content=baseline),
        ),
    )


def _task(role: DevelopmentRole) -> DevelopmentTask:
    now = datetime.now(UTC)
    return DevelopmentTask(
        id=1,
        feature_id=1,
        title="Task",
        description="Description.",
        assigned_role=role,
        status=TaskStatus.VERIFICATION_PENDING,
        created_at=now,
        updated_at=now,
    )


def _handoff(
    role: DevelopmentRole,
    changed_paths: tuple[str, ...],
) -> TaskHandoff:
    now = datetime.now(UTC)
    return TaskHandoff(
        id=1,
        task_id=1,
        agent_run_id=1,
        submitted_by=role,
        attribution=f"agent:{role.value}",
        workspace_identity_hash="workspace-hash",
        implementation_summary="Submitted implementation.",
        changed_paths=changed_paths,
        reused_symbols=(),
        new_symbols=(),
        reuse_notes="Used existing code where relevant.",
        checks_attempted=(
            "backend" if role.value.startswith("backend") else "frontend",
        ),
        limitations="none",
        next_action="verify",
        created_at=now,
    )


def _write(root: Path, path: str, content: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _failed_check_names(result: TaskVerificationResult) -> list[str]:
    return [check.name for check in result.checks if check.exit_code != 0]
