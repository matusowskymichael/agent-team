"""Behavior regressions using the real developer submission golden cases."""

from dataclasses import replace
from pathlib import Path

import pytest

from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.infrastructure.evaluation.evaluation_task_verifier import (
    EvaluationTaskVerifier,
)

CORRECT_IMPLEMENTATIONS = (
    (
        "bd-dev-002",
        "class AuthService:\n"
        "    def __init__(self):\n"
        "        self.revoked_tokens = set()\n"
        "    def logout(self, token: str) -> bool:\n"
        "        if not token:\n"
        "            return False\n"
        "        self.revoked_tokens.add(token)\n"
        "        return True\n",
    ),
    (
        "bd-dev-002",
        "class AuthService:\n"
        "    def __init__(self):\n"
        "        self.revoked_tokens: list[str] = []\n"
        "    def logout(self, token):\n"
        "        if token:\n"
        "            self.revoked_tokens.append(token)\n"
        "        return bool(token)\n",
    ),
    (
        "bd-dev-004",
        "import json\n"
        "class AuditExportFormatter:\n"
        "    def format(self, events):\n"
        "        return json.dumps(events)\n",
    ),
    (
        "bd-dev-004",
        "from json import dumps as serialize\n"
        "class AuditExportFormatter:\n"
        "    def format(self, events):\n"
        "        result: str = serialize(events, sort_keys=True)\n"
        "        return result\n",
    ),
    (
        "fd-dev-002",
        "export function AccountMenu({ onLogout }) {\n"
        "  return <button onClick={onLogout}>Logout</button>\n"
        "}\n",
    ),
    (
        "fd-dev-002",
        "export function AccountMenu(props: { onLogout: () => void }) {\n"
        "  return (<div><button onClick={() => props.onLogout()}>"
        "Logout</button></div>);\n"
        "}\n",
    ),
    (
        "fd-dev-004",
        "export function EmptyState({ message }: { message: string }) {\n"
        "  return <p>{message}</p>;\n"
        "}\n",
    ),
    (
        "fd-dev-004",
        "export function EmptyState(props) {\n"
        "  return (<section><span>{props.message}</span></section>);\n"
        "}\n",
    ),
)
WRONG_IMPLEMENTATIONS = (
    (
        "bd-dev-002",
        "class AuthService:\n"
        "    def logout(self, token):\n"
        "        # revoke the active token\n"
        "        return bool(token)\n",
    ),
    (
        "bd-dev-002",
        "class AuthService:\n"
        "    def __init__(self):\n"
        "        self.revoked_tokens = set()\n"
        "    def logout(self, token):\n"
        "        if False:\n"
        "            self.revoked_tokens.add(token)\n"
        "        return True\n",
    ),
    ("bd-dev-004", "class AuditExportFormatter:\n    pass\n"),
    (
        "bd-dev-004",
        "class AuditExportFormatter:\n"
        "    def format(self, events):\n"
        "        return '[]'\n",
    ),
    (
        "fd-dev-002",
        "export function AccountMenu({ onLogout }) {\n"
        "  return <button>Logout</button>\n"
        "}\n",
    ),
    (
        "fd-dev-002",
        "export function AccountMenu({ onLogout }) {\n"
        "  return <button onClick={() => {}}>Logout</button>\n"
        "}\n",
    ),
    (
        "fd-dev-002",
        "export function AccountMenu({ onLogout }) {\n"
        "  return <button onClick={onLogout()}>Logout</button>\n"
        "}\n",
    ),
    (
        "fd-dev-002",
        "export function AccountMenu({ onLogout }) {\n"
        "  return <button disabled onClick={onLogout}>Logout</button>\n"
        "}\n",
    ),
    (
        "fd-dev-004",
        "export function EmptyState({ message }) { return null }\n",
    ),
    (
        "fd-dev-004",
        "export function EmptyState({ message }) {\n"
        "  return <p>message</p>\n"
        "}\n",
    ),
    (
        "fd-dev-004",
        "export function EmptyState({ message }) {\n"
        "  return <p hidden>{message}</p>\n"
        "}\n",
    ),
)
SUBMISSION_CASES = ("bd-dev-002", "bd-dev-004", "fd-dev-002", "fd-dev-004")


class TestEvaluationTaskVerifier:
    """Evaluate behavior while keeping expected probes outside workspaces."""

    @pytest.mark.parametrize(
        ("evaluation_case", "implementation"),
        CORRECT_IMPLEMENTATIONS,
        indirect=("evaluation_case",),
    )
    def test_correct_implementations_pass(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        implementation: str,
    ) -> None:
        """Verify actual behavior for reused and new symbols in both roles."""
        target = seeded_workspace / submitted_handoff.changed_paths[0]
        target.write_text(implementation, encoding="utf-8")

        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )

        assert result.outcome is TaskVerificationOutcome.PASSED
        role = evaluation_case.active_role.value.partition("_")[0]
        assert [check.name for check in result.checks] == [
            role,
            f"{role}:workspace-diff",
            f"{role}:behavior",
        ]
        assert all(check.exit_code == 0 for check in result.checks)

    @pytest.mark.parametrize(
        ("evaluation_case", "implementation"),
        WRONG_IMPLEMENTATIONS,
        indirect=("evaluation_case",),
    )
    def test_wrong_implementations_fail_behavior(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        implementation: str,
    ) -> None:
        """Reject keywords, pass-only classes, dead code and inert handlers."""
        target = seeded_workspace / submitted_handoff.changed_paths[0]
        target.write_text(implementation, encoding="utf-8")

        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )

        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[-1].name.endswith(":behavior")
        assert result.checks[-1].exit_code == 1

    @pytest.mark.parametrize(
        "evaluation_case",
        SUBMISSION_CASES,
        indirect=True,
    )
    def test_no_op_workspace_fails(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
    ) -> None:
        """Never complete an unchanged or missing required implementation."""
        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )

        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[1].exit_code == 1

    @pytest.mark.parametrize(
        "evaluation_case",
        ("bd-dev-004",),
        indirect=True,
    )
    def test_unknown_case_blocks_without_inference_from_prompt(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
    ) -> None:
        """An unregistered case needs an explicit trusted behavior probe."""
        case = replace(evaluation_case, id="unknown-holdout-case")
        result = EvaluationTaskVerifier(case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )

        assert result.outcome is TaskVerificationOutcome.BLOCKED
        assert result.checks == ()


class TestEvaluationBehaviorBoundary:
    """Fail closed for syntax whose real semantics are not interpreted."""

    @pytest.mark.parametrize("evaluation_case", ("bd-dev-002",), indirect=True)
    @pytest.mark.parametrize(
        "source",
        (
            CORRECT_IMPLEMENTATIONS[0][1].replace("set()", "()"),
            CORRECT_IMPLEMENTATIONS[0][1].replace(
                "    def logout",
                "class AuthService:\n    def logout",
            ),
            "import json as bool\n" + CORRECT_IMPLEMENTATIONS[1][1],
            CORRECT_IMPLEMENTATIONS[0][1] + "import json as AuthService\n",
            CORRECT_IMPLEMENTATIONS[0][1].replace(
                "    def logout",
                "        return 1\n    def logout",
            ),
            CORRECT_IMPLEMENTATIONS[0][1].replace(
                "    def logout",
                "        self.logout = None\n    def logout",
            ),
            CORRECT_IMPLEMENTATIONS[0][1]
            + "    def __getattribute__(self, name):\n        return None\n",
            CORRECT_IMPLEMENTATIONS[0][1]
            + "    def __new__(self):\n        return None\n",
            CORRECT_IMPLEMENTATIONS[0][1]
            + "    def unused(self, value=print('not run')):\n        pass\n",
            CORRECT_IMPLEMENTATIONS[0][1].replace(
                "        if not token:\n            return False",
                "        if not token:\n"
                "            self.revoked_tokens.add(token)\n"
                "            return False",
            ),
            "# AuthService.logout revoked_tokens\n",
            "class AuthService:\n    def logout(self, token):\n"
            "        while True:\n            pass\n",
            CORRECT_IMPLEMENTATIONS[0][1].replace(
                "logout(self, token: str)",
                "logout(self, self: str)",
            ),
            "class AuthService:\n"
            "    def logout(self, token):\n"
            "        if not self:\n"
            "            self.revoked_tokens = []\n"
            "        if not token:\n"
            "            return False\n"
            "        self.revoked_tokens.append(token)\n"
            "        return True\n",
            CORRECT_IMPLEMENTATIONS[1][1] + "        bool = None\n",
        ),
        ids=(
            "tuple-is-immutable",
            "duplicate-class",
            "shadow-builtin",
            "shadow-class",
            "nonempty-initializer",
            "shadow-method",
            "custom-attribute-lookup",
            "custom-construction",
            "default-side-effect",
            "revokes-empty-token",
            "comment-only-symbol",
            "unbounded-loop",
            "duplicate-arguments",
            "instance-truth-value",
            "unreachable-local-shadow",
        ),
    )
    def test_unsupported_python_semantics_fail(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        source: str,
    ) -> None:
        """Reject sources that would fail real Python behavior probes."""
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_text(
            source,
            encoding="utf-8",
        )
        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[-1].exit_code == 1

    @pytest.mark.parametrize("evaluation_case", ("fd-dev-002",), indirect=True)
    @pytest.mark.parametrize(
        "jsx",
        (
            '<button onClick="evaluation-callback-reference">Logout</button>',
            '<button onClick="evaluation&#45;callback-reference">'
            "Logout</button>",
            "<button onClick={() => { onLogout(); onLogout(); }}></button>",
            "<button onClick={() => { return; onLogout(); }}>Logout</button>",
            '<button style="display:none" onClick={onLogout}>Logout</button>',
            '<button className="hidden" onClick={onLogout}>Logout</button>',
            "<Hidden><button onClick={onLogout}>Logout</button></Hidden>",
            "null; /* <button onClick={onLogout}>Logout</button> */",
            "<!DOCTYPE button [<!ENTITY text 'Logout'>]>"
            "<button onClick={onLogout}>&text;</button>",
            "<button onClick={onLogout}>Logout</span>",
            "<div onClick={onLogout}><button onClick={onLogout}>"
            "Logout</button></div>",
            '<?xml version="1.0"?><button onClick={onLogout}>Logout</button>',
        ),
        ids=(
            "literal-marker",
            "encoded-marker",
            "double-callback",
            "unreachable-callback",
            "hidden-style",
            "hidden-class",
            "unknown-wrapper",
            "comment-only-markers",
            "xml-entity",
            "invalid-jsx",
            "callback-bubbling",
            "xml-processing-instruction",
        ),
    )
    def test_unsupported_tsx_semantics_fail(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        jsx: str,
    ) -> None:
        """Preserve actual callback provenance and visible render behavior."""
        source = "export function AccountMenu({ onLogout }) { return " + (
            jsx + "; }"
        )
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_text(
            source,
            encoding="utf-8",
        )
        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[-1].exit_code == 1

    @pytest.mark.parametrize("evaluation_case", ("fd-dev-002",), indirect=True)
    @pytest.mark.parametrize(
        "source",
        (
            CORRECT_IMPLEMENTATIONS[4][1].replace(
                "{ onLogout }",
                "{ onLogout, onLogout }",
            ),
            CORRECT_IMPLEMENTATIONS[4][1].replace("return ", "return\n"),
        ),
        ids=("duplicate-props", "automatic-semicolon-insertion"),
    )
    def test_invalid_component_binding_or_return_fails(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        source: str,
    ) -> None:
        """Require JavaScript-valid bindings and a returned JSX value."""
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_text(
            source,
            encoding="utf-8",
        )
        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[-1].exit_code == 1

    @pytest.mark.parametrize("evaluation_case", ("bd-dev-004",), indirect=True)
    @pytest.mark.parametrize(
        "content",
        (b"\xff\xfe", b"x" * 20_001, b"class : invalid", b"x = 1\n" * 300),
        ids=(
            "invalid-utf8",
            "oversized-source",
            "invalid-source",
            "node-budget",
        ),
    )
    def test_invalid_or_unbounded_source_fails(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        content: bytes,
    ) -> None:
        """Bound file reads and AST walks while retaining failure evidence."""
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_bytes(
            content,
        )
        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[-1].exit_code == 1

    @pytest.mark.parametrize("evaluation_case", ("bd-dev-004",), indirect=True)
    def test_symlink_escape_is_not_read(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
    ) -> None:
        """A valid implementation outside the candidate root cannot pass."""
        outside = seeded_workspace.parent / "outside-formatter.py"
        outside.write_text(CORRECT_IMPLEMENTATIONS[2][1], encoding="utf-8")
        (seeded_workspace / submitted_handoff.changed_paths[0]).symlink_to(
            outside,
        )
        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[1].exit_code == 1
        assert result.checks[-1].exit_code == 1

    @pytest.mark.parametrize("evaluation_case", ("bd-dev-004",), indirect=True)
    def test_missing_handoff_path_fails(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
    ) -> None:
        """Correct code still requires the expected submitted changed path."""
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_text(
            CORRECT_IMPLEMENTATIONS[2][1],
            encoding="utf-8",
        )
        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            replace(submitted_handoff, changed_paths=()),
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[0].exit_code == 1
        assert result.checks[-1].exit_code == 0

    @pytest.mark.parametrize("evaluation_case", ("bd-dev-004",), indirect=True)
    def test_missing_expected_paths_blocks(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
    ) -> None:
        """Legacy loading remains valid but absent expectations fail closed."""
        result = EvaluationTaskVerifier(
            replace(evaluation_case, expected_tool_calls=()),
        ).verify(verification_task, submitted_handoff, seeded_workspace)
        assert result.outcome is TaskVerificationOutcome.BLOCKED
        assert result.checks == ()

    @pytest.mark.parametrize("evaluation_case", ("bd-dev-004",), indirect=True)
    @pytest.mark.parametrize("path", ("../escape.py", "/escape.py", ""))
    def test_invalid_expected_path_fails_closed(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        path: str,
    ) -> None:
        """Dataset path assertions cannot escape the workspace boundary."""
        case = replace(
            evaluation_case,
            expected_tool_calls=(
                ExpectedToolCall("apply_patch", {"path": path}),
            ),
        )
        result = EvaluationTaskVerifier(case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[1].exit_code == 1

    @pytest.mark.parametrize("evaluation_case", ("bd-dev-004",), indirect=True)
    def test_alternative_trajectory_retains_path_gate(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
    ) -> None:
        """Read path assertions from accepted legacy trajectories."""
        target = seeded_workspace / submitted_handoff.changed_paths[0]
        target.write_text("   ", encoding="utf-8")
        case = replace(
            evaluation_case,
            expected_tool_calls=(
                ExpectedToolCall("apply_patch", {"path": 1}),
            ),
            acceptable_tool_trajectories=(
                ExpectedToolTrajectory(
                    required_tool_calls=evaluation_case.expected_tool_calls,
                ),
            ),
        )
        result = EvaluationTaskVerifier(case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[1].exit_code == 1

    @pytest.mark.parametrize("evaluation_case", ("bd-dev-004",), indirect=True)
    def test_candidate_source_has_no_io_capabilities(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
    ) -> None:
        """Reject imports rather than running dataset or candidate commands."""
        marker = seeded_workspace / "candidate-must-not-write"
        source = (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            + CORRECT_IMPLEMENTATIONS[2][1]
        )
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_text(
            source,
            encoding="utf-8",
        )
        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task,
            submitted_handoff,
            seeded_workspace,
        )
        assert result.outcome is TaskVerificationOutcome.FAILED
        assert not marker.exists()
