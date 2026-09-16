"""Authoritative task state used between execution segments."""

import json
from dataclasses import dataclass

from agent_team.application.audit.audit_sanitizer import (
    hash_text,
    sanitize_text,
)
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_verification_evidence import (
    TaskVerificationEvidence,
)


@dataclass(frozen=True, slots=True)
class AgentTaskSnapshot:
    """A task and its latest persisted handoff and verification."""

    task: DevelopmentTask
    handoff: TaskHandoff | None
    verification: TaskVerificationEvidence | None

    def fingerprint(self) -> str:
        """Identify semantic state without row IDs or timestamps."""
        handoff = self.handoff
        verification = self.verification
        values = {
            "status": self.task.status.value,
            "handoff": None
            if handoff is None
            else (
                handoff.changed_paths,
                handoff.checks_attempted,
            ),
            "verification": None
            if verification is None
            else (
                verification.outcome.value,
                verification.failure_classification.value,
                tuple(
                    (
                        check.name,
                        check.exit_code,
                        check.timed_out,
                    )
                    for check in verification.checks
                ),
            ),
        }
        return hash_text(json.dumps(values, sort_keys=True))

    def render(self) -> str:
        """Render bounded task facts without internal workspace identity."""
        lines = [
            f"Current task {self.task.id} status: {self.task.status.value}.",
        ]
        if self.handoff is not None:
            lines.append(f"Latest persisted submission: {self.handoff.id}.")
            lines.append(
                "Latest handoff summary: "
                f"{sanitize_text(self.handoff.implementation_summary)}",
            )
        if self.verification is not None:
            lines.extend(
                (
                    "Verification submission: "
                    f"{self.verification.submission_id}.",
                    f"Verification result: {self.verification.outcome.value}.",
                    "Verification feedback: "
                    f"{sanitize_text(self.verification.feedback)}",
                )
            )
        return "\n".join(lines)
