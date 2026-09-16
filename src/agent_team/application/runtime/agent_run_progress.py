"""Objective progress detection and bounded continuation ledger."""

import json
from dataclasses import dataclass, field
from typing import cast

from agent_team.application.audit.audit_sanitizer import (
    hash_text,
    sanitize_text,
)
from agent_team.application.runtime.agent_task_snapshot import (
    AgentTaskSnapshot,
)
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_stalled_error import AgentStalledError
from agent_team.domain.workflow.task_status import TaskStatus

MAX_LEDGER_ENTRIES = 16
MAX_CHANGED_PATHS = 24
MAX_COMPLETED_CHECKS = 12


@dataclass(slots=True)
class AgentRunProgress:
    """Track unique observed work, with no cap on legitimate operations."""

    max_no_progress_segments: int
    segment_count: int = 0
    turns_used: int = 0
    mutation_started: bool = False
    no_progress_segments: int = 0
    _fingerprints: set[str] = field(default_factory=set[str])
    _observed_ids: set[int] = field(default_factory=set[int])
    _ledger: list[str] = field(default_factory=list[str])
    _changed_paths: set[str] = field(default_factory=set[str])
    _completed_checks: dict[str, str] = field(default_factory=dict[str, str])
    _last_status: TaskStatus | None = None
    _patch_revision: int = 0

    def seed(self, snapshot: AgentTaskSnapshot | None) -> None:
        """Exclude preexisting workflow state from newly observed progress."""
        if snapshot is not None:
            self._fingerprints.add(snapshot.fingerprint())
            self._last_status = snapshot.task.status
            self.mutation_started = (
                snapshot.task.status is TaskStatus.VERIFICATION_PENDING
            )

    def observe(
        self,
        invocations: list[ToolInvocationRecord],
        snapshot: AgentTaskSnapshot | None,
    ) -> None:
        """Count new successful facts without IDs, denials or narration."""
        previous_count = len(self._fingerprints)
        for invocation in sorted(invocations, key=lambda item: item.id):
            if invocation.id in self._observed_ids:
                continue
            if invocation.status is not ToolInvocationStatus.ALLOWED:
                self._observed_ids.add(invocation.id)
            self._observe_invocation(invocation)
        if snapshot is not None:
            if snapshot.task.status is not self._last_status and (
                snapshot.task.status is TaskStatus.IN_PROGRESS
            ):
                self.mutation_started = True
            self._last_status = snapshot.task.status
            self._fingerprints.add(snapshot.fingerprint())
        if len(self._fingerprints) > previous_count:
            self.no_progress_segments = 0
        else:
            self.no_progress_segments += 1

    def require_progress(self) -> None:
        """Raise only after consecutive complete segments without new work."""
        if self.no_progress_segments >= self.max_no_progress_segments:
            raise AgentStalledError(
                "Agent stalled: "
                f"{self.no_progress_segments} consecutive segments produced "
                "no new successful operations or authoritative state changes.",
            )

    def continuation_context(self, snapshot: AgentTaskSnapshot | None) -> str:
        """Compact to trusted facts rather than replaying prior model turns."""
        omitted = max(0, len(self._ledger) - MAX_LEDGER_ENTRIES)
        lines = [
            "Continue the same logical run and original user request.",
            "Feature, task, role, workspace and session bindings "
            "are unchanged.",
            "Already-successful operations must not be replayed. Inspect "
            "current state and perform only the next unfinished action.",
            "Do not repeat probes of directories already known to be absent.",
            "After an aggregate check passes, immediately submit the handoff.",
            "If verification failed, inspect its feedback, repair, check and "
            "resubmit. Finish at completed or validly blocked task state.",
            f"Internal segments executed: {self.segment_count}.",
            *_summary_lines(
                "Successful changed paths",
                sorted(self._changed_paths),
                MAX_CHANGED_PATHS,
            ),
            *_summary_lines(
                "Completed trusted checks",
                [
                    self._completed_checks[name]
                    for name in sorted(self._completed_checks)
                ],
                MAX_COMPLETED_CHECKS,
            ),
            f"Successful-operation ledger: {len(self._ledger)} "
            "unique entries; "
            f"{omitted} older entries omitted from this bounded view.",
            *self._ledger[-MAX_LEDGER_ENTRIES:],
        ]
        if snapshot is not None:
            lines.append(snapshot.render())
        return "\n".join(lines)

    def _observe_invocation(self, invocation: ToolInvocationRecord) -> None:
        mutating = invocation.classification is ToolClassification.MUTATING
        if mutating and invocation.status is not ToolInvocationStatus.DENIED:
            self.mutation_started = True
        if invocation.status is not ToolInvocationStatus.COMPLETED:
            return
        result = _result_object(invocation.result_preview)
        fingerprint = _operation_fingerprint(
            invocation, result, self._patch_revision
        )
        if fingerprint is None:
            return
        self._remember_result(invocation.tool_name, result)
        if fingerprint in self._fingerprints:
            return
        self._fingerprints.add(fingerprint)
        if invocation.tool_name == "apply_patch":
            self._patch_revision += 1
        target = result.get("path", result.get("name", ""))
        self._ledger.append(
            f"- {sanitize_text(invocation.tool_name)} "
            f"{sanitize_text(target)} succeeded; "
            f"arguments={sanitize_text(invocation.arguments_hash)}; "
            f"result={sanitize_text(invocation.result_hash)}.",
        )

    def _remember_result(
        self,
        tool_name: str,
        result: dict[str, object],
    ) -> None:
        if tool_name == "apply_patch":
            state = _patch_state(result)
            if state is not None:
                self._changed_paths.add(state[0])
        elif tool_name == "run_check":
            check = _check_state(result)
            if check is not None:
                name, exit_code, timed_out = check
                self._completed_checks[name] = (
                    f"{sanitize_text(name)}; exit_code={exit_code}; "
                    f"timed_out={timed_out}"
                )


def _operation_fingerprint(
    invocation: ToolInvocationRecord,
    result: dict[str, object],
    patch_revision: int,
) -> str | None:
    state: object
    if invocation.tool_name == "apply_patch":
        state = _patch_state(result)
        if state is None:
            return None
    elif invocation.tool_name == "run_check":
        check = _check_state(result)
        if check is None:
            return None
        state = check, patch_revision
    elif invocation.tool_name == "submit_task_for_verification":
        arguments = _result_object(invocation.arguments_preview_json)
        state = (
            arguments.get("task_id"),
            _sorted_strings(arguments.get("changed_paths")),
            _sorted_strings(arguments.get("checks_attempted")),
        )
    else:
        result_key = (
            invocation.result_hash
            if invocation.classification is ToolClassification.READ_ONLY
            else None
        )
        state = invocation.arguments_hash, result_key
    return hash_text(
        json.dumps(
            (
                invocation.server_name,
                invocation.tool_name,
                state,
            )
        )
    )


def _patch_state(result: dict[str, object]) -> tuple[str, str] | None:
    path = result.get("path")
    after_hash = result.get("after_hash")
    if (
        result.get("applied") is not True
        or not isinstance(path, str)
        or not path.strip()
        or not isinstance(after_hash, str)
        or not after_hash
        or result.get("before_hash") == after_hash
    ):
        return None
    return path, after_hash


def _check_state(result: dict[str, object]) -> tuple[str, int, bool] | None:
    name = result.get("name")
    exit_code = result.get("exit_code")
    timed_out = result.get("timed_out")
    if (
        not isinstance(name, str)
        or not name.strip()
        or type(exit_code) is not int
        or not isinstance(timed_out, bool)
    ):
        return None
    return name, exit_code, timed_out


def _sorted_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    values = cast("list[object] | tuple[object, ...]", value)
    return tuple(sorted({item for item in values if isinstance(item, str)}))


def _summary_lines(label: str, entries: list[str], limit: int) -> list[str]:
    omitted = max(0, len(entries) - limit)
    return [
        f"{label}: {len(entries)} unique entries; {omitted} omitted.",
        *(f"- {sanitize_text(entry)}" for entry in entries[:limit]),
    ]


def _result_object(value: str | None) -> dict[str, object]:
    if value is None:
        return {}
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return (
        cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}
    )
