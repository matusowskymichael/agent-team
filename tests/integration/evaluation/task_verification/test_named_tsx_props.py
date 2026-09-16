"""Named local TSX prop regressions against unchanged behavior assertions."""

from itertools import product
from pathlib import Path

import pytest

from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.infrastructure.evaluation.evaluation_task_verifier import (
    EvaluationTaskVerifier,
)

LOCAL_PROP_DECLARATIONS = (
    "interface {name} {{ {field}; }}\n",
    "export interface {name} {{ {field}; }}\n",
    "type {name} = {{ {field}; }};\n",
    "export type {name} = {{ {field}; }};\n",
)


class TestNamedTsxProps:
    """Require real callback and visible-text behavior for local prop types."""

    @pytest.mark.parametrize("evaluation_case", ("fd-dev-002",), indirect=True)
    @pytest.mark.parametrize(
        "definition",
        tuple(
            product(
                LOCAL_PROP_DECLARATIONS,
                (
                    ("{ onLogout }", "onLogout"),
                    ("props", "() => props.onLogout()"),
                ),
            )
        ),
    )
    def test_named_callback_props_pass_hidden_behavior(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        definition: tuple[str, tuple[str, str]],
    ) -> None:
        """Accept local interfaces and aliases with correct logout binding."""
        declaration, (parameter, callback) = definition
        source = declaration.format(
            name="AccountMenuProps", field="onLogout: () => void"
        ) + (
            f"export function AccountMenu({parameter}: AccountMenuProps) {{\n"
            f'  return (<button type="button" onClick={{{callback}}}>'
            "Logout</button>);\n}\n"
        )
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_text(
            source, encoding="utf-8"
        )

        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task, submitted_handoff, seeded_workspace
        )

        assert result.outcome is TaskVerificationOutcome.PASSED
        assert result.checks[-1].name == "frontend:behavior"
        assert result.checks[-1].exit_code == 0

    @pytest.mark.parametrize("evaluation_case", ("fd-dev-004",), indirect=True)
    @pytest.mark.parametrize("declaration", LOCAL_PROP_DECLARATIONS)
    def test_named_string_props_pass_hidden_behavior(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        declaration: str,
    ) -> None:
        """Retain supplied visible text through a named local string shape."""
        source = declaration.format(
            name="EmptyStateProps", field="message: string"
        ) + (
            "export function EmptyState({ message }: EmptyStateProps) {\n"
            "  return <p>{message}</p>;\n}\n"
        )
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_text(
            source, encoding="utf-8"
        )

        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task, submitted_handoff, seeded_workspace
        )

        assert result.outcome is TaskVerificationOutcome.PASSED
        assert result.checks[-1].exit_code == 0

    @pytest.mark.parametrize("evaluation_case", ("fd-dev-002",), indirect=True)
    @pytest.mark.parametrize(
        "definition",
        tuple(
            product(
                LOCAL_PROP_DECLARATIONS,
                (
                    "<button>Logout</button>",
                    "<button onClick={() => {}}>Logout</button>",
                    "<button onClick={onLogout()}>Logout</button>",
                    "<button onClick={() => { onLogout(); onLogout(); }}>"
                    "Logout</button>",
                    "<button disabled onClick={onLogout}>Logout</button>",
                    "<button hidden onClick={onLogout}>Logout</button>",
                    '<button className="hidden" onClick={onLogout}>'
                    "Logout</button>",
                    "<button onClick={onLogout}>Login</button>",
                    "<CustomButton onClick={onLogout}>Logout</CustomButton>",
                ),
            )
        ),
    )
    def test_named_props_preserve_hidden_behavior_failures(
        self,
        evaluation_case: EvalCase,
        verification_task: DevelopmentTask,
        submitted_handoff: TaskHandoff,
        seeded_workspace: Path,
        definition: tuple[str, str],
    ) -> None:
        """Type syntax cannot bypass callback, visibility or text checks."""
        declaration, jsx = definition
        source = declaration.format(
            name="AccountMenuProps", field="onLogout: () => void"
        ) + (
            "export function AccountMenu({ onLogout }: AccountMenuProps) {\n"
            f"  return {jsx};\n}}\n"
        )
        (seeded_workspace / submitted_handoff.changed_paths[0]).write_text(
            source, encoding="utf-8"
        )

        result = EvaluationTaskVerifier(evaluation_case).verify(
            verification_task, submitted_handoff, seeded_workspace
        )

        assert result.outcome is TaskVerificationOutcome.FAILED
        assert result.checks[-1].name == "frontend:behavior"
        assert result.checks[-1].exit_code == 1
