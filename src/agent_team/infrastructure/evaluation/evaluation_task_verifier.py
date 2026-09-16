"""Deterministic evaluation workspace verifier."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast
from xml.etree.ElementTree import ParseError

from agent_team.application.audit.audit_sanitizer import (
    hash_text,
    sanitize_text,
)
from agent_team.application.workflow.task_verifier import TaskVerifier
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_check_result import (
    TaskVerificationCheckResult,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)
from agent_team.infrastructure.evaluation.bounded_python_probe import (
    BoundedPythonProbe,
)
from agent_team.infrastructure.evaluation.bounded_tsx_probe import (
    CALLBACK_MARKER,
    probe_component,
)

VERIFIER_NAME = "local_workspace_checks"
MAX_ASSERTION_SOURCE_CHARS = 20_000
FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)


@dataclass(frozen=True, slots=True)
class EvaluationTaskVerifier(TaskVerifier):
    """Verify eval submissions with hidden deterministic assertions."""

    case: EvalCase

    def verify(
        self,
        task: DevelopmentTask,
        handoff: TaskHandoff,
        workspace_root: Path,
    ) -> TaskVerificationResult:
        """Assert expected workspace changes after an evaluation run."""
        started_at = _utc_now()
        expected_paths = _expected_patch_paths(self.case)
        if not expected_paths:
            return _blocked(
                started_at,
                (),
                "Evaluation case has no expected patch path assertions.",
            )

        if _behavior_contract(self.case) is None:
            return _blocked(
                started_at,
                (),
                "Evaluation case needs a registered behavior contract.",
            )

        checks = _verification_checks(
            self.case,
            task,
            handoff,
            workspace_root,
            expected_paths,
        )
        failure = next(
            (check for check in checks if check.exit_code != 0),
            None,
        )
        if failure is not None:
            return TaskVerificationResult(
                verifier_name=VERIFIER_NAME,
                outcome=TaskVerificationOutcome.FAILED,
                failure_classification=FailureClassification.CHECK_FAILED,
                feedback=(
                    f"Evaluation assertion {failure.name} failed. "
                    "Fix the workspace change and resubmit the task."
                ),
                checks=checks,
                started_at=started_at,
                ended_at=_utc_now(),
            )

        return TaskVerificationResult(
            verifier_name=VERIFIER_NAME,
            outcome=TaskVerificationOutcome.PASSED,
            failure_classification=FailureClassification.NONE,
            feedback="All required verification checks passed.",
            checks=checks,
            started_at=started_at,
            ended_at=_utc_now(),
        )


def _verification_checks(
    case: EvalCase,
    task: DevelopmentTask,
    handoff: TaskHandoff,
    workspace_root: Path,
    expected_paths: tuple[str, ...],
) -> tuple[TaskVerificationCheckResult, ...]:
    check_name = _aggregate_check_name(task)
    checks = [
        _check(
            name=check_name,
            passed=_has_expected_handoff_paths(handoff, expected_paths),
            success_text="Expected changed paths were submitted.",
            failure_text="Submitted handoff did not include expected paths.",
        ),
        _check(
            name=f"{check_name}:workspace-diff",
            passed=_expected_paths_changed(
                case,
                workspace_root,
                expected_paths,
            ),
            success_text="Expected workspace paths changed.",
            failure_text="Expected workspace paths were unchanged or missing.",
        ),
    ]
    checks.append(
        _check(
            name=f"{check_name}:behavior",
            passed=(
                task.assigned_role is case.active_role
                and _behavior_passes(case, workspace_root)
            ),
            success_text="Required implementation behavior passed.",
            failure_text="Required implementation behavior did not pass.",
        ),
    )
    return tuple(checks)


def _blocked(
    started_at: datetime,
    checks: tuple[TaskVerificationCheckResult, ...],
    feedback: str,
) -> TaskVerificationResult:
    return TaskVerificationResult(
        verifier_name=VERIFIER_NAME,
        outcome=TaskVerificationOutcome.BLOCKED,
        failure_classification=FailureClassification.CONFIGURATION_ERROR,
        feedback=feedback,
        checks=checks,
        started_at=started_at,
        ended_at=_utc_now(),
    )


def _check(
    name: str,
    passed: bool,
    success_text: str,
    failure_text: str,
) -> TaskVerificationCheckResult:
    timestamp = _utc_now()
    stdout_excerpt = sanitize_text(success_text if passed else "")
    stderr_excerpt = sanitize_text("" if passed else failure_text)
    return TaskVerificationCheckResult(
        name=name,
        started_at=timestamp,
        ended_at=timestamp,
        exit_code=0 if passed else 1,
        timed_out=False,
        stdout_hash=hash_text(stdout_excerpt),
        stdout_excerpt=stdout_excerpt,
        stderr_hash=hash_text(stderr_excerpt),
        stderr_excerpt=stderr_excerpt,
    )


def _aggregate_check_name(task: DevelopmentTask) -> str:
    if task.assigned_role.value.startswith("frontend"):
        return "frontend"
    return "backend"


def _has_expected_handoff_paths(
    handoff: TaskHandoff,
    expected_paths: tuple[str, ...],
) -> bool:
    return set(expected_paths).issubset(set(handoff.changed_paths))


def _expected_paths_changed(
    case: EvalCase,
    workspace_root: Path,
    expected_paths: tuple[str, ...],
) -> bool:
    for path in expected_paths:
        current = _read_workspace_file(workspace_root, path)
        baseline = _baseline_content(case, path)
        if current is None:
            return False
        if baseline is not None and current == baseline:
            return False
        if baseline is None and not current.strip():
            return False
    return True


def _read_workspace_file(
    workspace_root: Path,
    path: str,
) -> str | None:
    safe_path = _safe_relative_path(path)
    if safe_path is None:
        return None
    try:
        root = workspace_root.resolve(strict=True)
        target = (root / safe_path).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            return None
        with target.open(encoding="utf-8") as handle:
            source = handle.read(MAX_ASSERTION_SOURCE_CHARS + 1)
        return source if len(source) <= MAX_ASSERTION_SOURCE_CHARS else None
    except OSError, ValueError, UnicodeError:
        return None


def _safe_relative_path(path: str) -> Path | None:
    pure_path = PurePosixPath(path.replace("\\", "/"))
    if pure_path.is_absolute() or ".." in pure_path.parts:
        return None
    normalized = pure_path.as_posix().strip("/")
    if normalized in {"", "."}:
        return None
    return Path(normalized)


def _baseline_content(case: EvalCase, path: str) -> str | None:
    normalized = path.replace("\\", "/").strip("/")
    for file_fixture in case.workspace_files:
        if file_fixture.path.replace("\\", "/").strip("/") == normalized:
            return file_fixture.content
    return None


def _expected_patch_paths(case: EvalCase) -> tuple[str, ...]:
    paths: list[str] = []
    for call in _expected_calls(case):
        if call.name != "apply_patch":
            continue
        path = call.arguments_subset.get("path")
        if isinstance(path, str):
            paths.append(path)
    return tuple(dict.fromkeys(paths))


def _expected_calls(case: EvalCase) -> tuple[ExpectedToolCall, ...]:
    calls = list(case.expected_tool_calls)
    for trajectory in case.acceptable_tool_trajectories:
        calls.extend(trajectory.required_tool_calls)
    return tuple(calls)


def _behavior_contract(
    case: EvalCase,
) -> tuple[str, Callable[[str], bool]] | None:
    # Trusted runner code is the allow-list: datasets cannot supply code,
    # commands or probe inputs. New/holdout cases require explicit review.
    contracts: dict[str, tuple[str, Callable[[str], bool]]] = {
        "bd-dev-002": ("backend/auth_service.py", _check_logout),
        "bd-dev-004": ("backend/audit_export.py", _check_audit_export),
        "fd-dev-002": ("frontend/AccountMenu.tsx", _check_account_menu),
        "fd-dev-004": ("frontend/EmptyState.tsx", _check_empty_state),
    }
    return contracts.get(case.id)


def _behavior_passes(case: EvalCase, workspace_root: Path) -> bool:
    contract = _behavior_contract(case)
    if contract is None:
        return False
    path, assertion = contract
    source = _read_workspace_file(workspace_root, path)
    if source is None:
        return False
    try:
        return assertion(source)
    except (
        SyntaxError,
        ValueError,
        KeyError,
        TypeError,
        RecursionError,
        ParseError,
    ):
        # Unsupported candidate syntax is a deterministic failed assertion.
        # No source, probe values or expected output enters model evidence.
        return False


def _check_logout(source: str) -> bool:
    probe = BoundedPythonProbe(source, "AuthService")
    if probe.call("logout", ("",)) is not False:
        return False
    expected: set[str] = set()
    for token in ("active-session-17", "other-session-29"):
        if probe.call("logout", (token,)) is not True:
            return False
        expected.add(token)
        revoked = probe.state.get("revoked_tokens")
        if (
            not isinstance(revoked, (list, set))
            or set(cast("list[object] | set[object]", revoked)) != expected
        ):
            return False
    if probe.call("logout", ("",)) is not False:
        return False
    revoked = probe.state.get("revoked_tokens")
    return (
        isinstance(revoked, (list, set))
        and set(cast("list[object] | set[object]", revoked)) == expected
    )


def _check_audit_export(source: str) -> bool:
    probe = BoundedPythonProbe(source, "AuditExportFormatter")
    samples: tuple[list[dict[str, object]], ...] = (
        [],
        [{"event": "login", "actor": "Zoë", "count": 2}],
        [{"event": "logout", "metadata": {"success": True}}, {"id": 17}],
    )
    for events in samples:
        expected = json.dumps(events)
        result = probe.call("format", (events,))
        if not isinstance(result, str) or json.loads(result) != json.loads(
            expected,
        ):
            return False
        if json.dumps(events) != expected:
            return False
    return True


def _check_account_menu(source: str) -> bool:
    rendered = probe_component(
        source,
        "AccountMenu",
        {"onLogout": CALLBACK_MARKER},
    )
    buttons = [
        element
        for element in rendered.iter("button")
        if "".join(element.itertext()).strip().casefold() == "logout"
    ]
    callbacks = [
        element
        for element in rendered.iter()
        if element.get("onClick") is not None
    ]
    return (
        len(buttons) == 1
        and callbacks == buttons
        and buttons[0].get("onClick") == CALLBACK_MARKER
    )


def _check_empty_state(source: str) -> bool:
    for message in ("No invoices yet", "Nothing for Zoë & team"):
        rendered = probe_component(source, "EmptyState", {"message": message})
        if message not in "".join(rendered.itertext()):
            return False
    return True


def _utc_now() -> datetime:
    return datetime.now(UTC)
