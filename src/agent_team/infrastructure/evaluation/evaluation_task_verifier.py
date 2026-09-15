"""Deterministic evaluation workspace verifier."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

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

VERIFIER_NAME = "local_workspace_checks"
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
    symbols = _expected_symbols(case)
    if symbols:
        checks.append(
            _check(
                name=f"{check_name}:symbols",
                passed=_symbols_present(
                    workspace_root,
                    expected_paths,
                    symbols,
                ),
                success_text="Expected symbols are present.",
                failure_text="Expected symbols were not present.",
            ),
        )
    markers = _expected_markers(case)
    if markers:
        checks.append(
            _check(
                name=f"{check_name}:behavior-markers",
                passed=_markers_present(
                    workspace_root,
                    expected_paths,
                    markers,
                ),
                success_text="Expected behavior markers are present.",
                failure_text="Expected behavior markers were not present.",
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


def _symbols_present(
    workspace_root: Path,
    expected_paths: tuple[str, ...],
    symbols: tuple[str, ...],
) -> bool:
    combined = _combined_content(workspace_root, expected_paths)
    return all(_symbol_name(symbol) in combined for symbol in symbols)


def _markers_present(
    workspace_root: Path,
    expected_paths: tuple[str, ...],
    markers: tuple[str, ...],
) -> bool:
    combined = _combined_content(workspace_root, expected_paths).casefold()
    return all(marker.casefold() in combined for marker in markers)


def _combined_content(
    workspace_root: Path,
    expected_paths: tuple[str, ...],
) -> str:
    return "\n".join(
        content
        for path in expected_paths
        if (content := _read_workspace_file(workspace_root, path)) is not None
    )


def _read_workspace_file(
    workspace_root: Path,
    path: str,
) -> str | None:
    safe_path = _safe_relative_path(path)
    if safe_path is None:
        return None
    target = workspace_root / safe_path
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8")
    except OSError:
        return None


def _safe_relative_path(path: str) -> Path | None:
    pure_path = PurePosixPath(path.replace("\\", "/"))
    if pure_path.is_absolute() or ".." in pure_path.parts:
        return None
    normalized = pure_path.as_posix().strip("/")
    if not normalized:
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


def _expected_symbols(case: EvalCase) -> tuple[str, ...]:
    symbols: list[str] = []
    for call in _expected_calls(case):
        if call.name != "find_symbol":
            continue
        name = call.arguments_subset.get("name")
        if isinstance(name, str):
            symbols.append(name)
    return tuple(dict.fromkeys(symbols))


def _expected_calls(case: EvalCase) -> tuple[ExpectedToolCall, ...]:
    calls = list(case.expected_tool_calls)
    for trajectory in case.acceptable_tool_trajectories:
        calls.extend(trajectory.required_tool_calls)
    return tuple(calls)


def _expected_markers(case: EvalCase) -> tuple[str, ...]:
    prompt = _case_text(case).casefold()
    markers: list[str] = []
    if "revok" in prompt:
        markers.append("revok")
    if "onlogout" in prompt:
        markers.append("onLogout")
    return tuple(markers)


def _case_text(case: EvalCase) -> str:
    task_descriptions = [
        task.description
        for feature in case.feature_fixtures
        for task in feature.tasks
    ]
    return "\n".join((case.user_input, *task_descriptions))


def _symbol_name(symbol: str) -> str:
    return symbol.rsplit(".", maxsplit=1)[-1]


def _utc_now() -> datetime:
    return datetime.now(UTC)
