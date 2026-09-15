"""SQLite-backed workflow repository."""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow import (
    task_verification_failure_classification as failure_classification,
)
from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_handoff import TaskHandoff
from agent_team.domain.workflow.task_handoff_draft import TaskHandoffDraft
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.domain.workflow.task_verification_check import (
    TaskVerificationCheck,
)
from agent_team.domain.workflow.task_verification_contract import (
    TaskVerificationContract,
)
from agent_team.domain.workflow.task_verification_evidence import (
    TaskVerificationEvidence,
)
from agent_team.domain.workflow.task_verification_outcome import (
    TaskVerificationOutcome,
)
from agent_team.domain.workflow.task_verification_profiles import (
    default_verification_contract,
)
from agent_team.domain.workflow.task_verification_result import (
    TaskVerificationResult,
)
from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_schema_migrator,
)

FailureClassification = (
    failure_classification.TaskVerificationFailureClassification
)
_TASK_SELECT_SQL = """
SELECT
    id,
    feature_id,
    title,
    description,
    assigned_role,
    status,
    verification_profile,
    required_checks_json,
    created_at,
    updated_at
FROM development_tasks
"""
_STATUS_PLACEHOLDERS = "?, ?, ?, ?, ?"
_UNMATCHED_STATUS = "__unmatched_task_status__"


@dataclass(frozen=True, slots=True)
class SQLiteWorkflowRepository:
    """SQLite implementation of workflow persistence."""

    database_path: Path

    def __post_init__(self) -> None:
        """Create the database directory and migrate the schema."""
        sqlite_workflow_schema_migrator.SQLiteWorkflowSchemaMigrator(
            self.database_path,
        ).migrate()

    def create_feature(
        self,
        title: str,
        description: str,
        status: FeatureStatus,
    ) -> Feature:
        """Create and persist a feature."""
        timestamp = _utc_now()
        timestamp_text = _format_timestamp(timestamp)
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO features (
                    title,
                    description,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    title,
                    description,
                    status.value,
                    timestamp_text,
                    timestamp_text,
                ),
            )
            return _select_feature(connection, _last_insert_id(cursor))

    def get_feature(self, feature_id: int) -> Feature | None:
        """Return a feature by ID, if it exists."""
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT id, title, description, status, created_at, updated_at
                FROM features
                WHERE id = ?
                """,
                (feature_id,),
            ).fetchone()
            return None if row is None else _map_feature(row)

    def list_features(
        self,
        status: FeatureStatus | None = None,
    ) -> list[Feature]:
        """Return persisted features, optionally filtered by status."""
        with self._transaction() as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT
                        id,
                        title,
                        description,
                        status,
                        created_at,
                        updated_at
                    FROM features
                    ORDER BY id
                    """,
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        id,
                        title,
                        description,
                        status,
                        created_at,
                        updated_at
                    FROM features
                    WHERE status = ?
                    ORDER BY id
                    """,
                    (status.value,),
                ).fetchall()
            return [_map_feature(row) for row in rows]

    def add_artifact(
        self,
        feature_id: int,
        kind: ArtifactKind,
        content: str,
        created_by: str,
    ) -> Artifact:
        """Create and persist a feature artifact."""
        timestamp_text = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO artifacts (
                    feature_id,
                    kind,
                    content,
                    created_by,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feature_id,
                    kind.value,
                    content,
                    created_by,
                    timestamp_text,
                ),
            )
            return _select_artifact(connection, _last_insert_id(cursor))

    def list_artifacts(self, feature_id: int) -> list[Artifact]:
        """Return artifacts attached to a feature."""
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT id, feature_id, kind, content, created_by, created_at
                FROM artifacts
                WHERE feature_id = ?
                ORDER BY id
                """,
                (feature_id,),
            ).fetchall()
            return [_map_artifact(row) for row in rows]

    def create_task(
        self,
        feature_id: int,
        title: str,
        description: str,
        assigned_role: DevelopmentRole,
        status: TaskStatus,
    ) -> DevelopmentTask:
        """Create and persist a development task."""
        timestamp_text = _format_timestamp(_utc_now())
        contract = default_verification_contract(assigned_role)
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO development_tasks (
                    feature_id,
                    title,
                    description,
                    assigned_role,
                    status,
                    verification_profile,
                    required_checks_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feature_id,
                    title,
                    description,
                    assigned_role.value,
                    status.value,
                    None if contract is None else contract.profile_name,
                    _contract_checks_json(contract),
                    timestamp_text,
                    timestamp_text,
                ),
            )
            return _select_task(connection, _last_insert_id(cursor))

    def get_task(self, task_id: int) -> DevelopmentTask | None:
        """Return a development task by ID, if it exists."""
        with self._transaction() as connection:
            return _select_optional_task(connection, task_id)

    def list_tasks(self, feature_id: int) -> list[DevelopmentTask]:
        """Return development tasks attached to a feature."""
        with self._transaction() as connection:
            rows = connection.execute(
                f"""
                {_TASK_SELECT_SQL}
                WHERE feature_id = ?
                ORDER BY id
                """,
                (feature_id,),
            ).fetchall()
            return [_map_development_task(row) for row in rows]

    def update_task_status(
        self,
        task_id: int,
        status: TaskStatus,
    ) -> DevelopmentTask | None:
        """Update a task status and return the updated task, if it exists."""
        timestamp_text = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE development_tasks
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, timestamp_text, task_id),
            )
            return _select_optional_task(connection, task_id)

    def claim_task_for_work(
        self,
        task_id: int,
        from_statuses: frozenset[TaskStatus],
        active_statuses: frozenset[TaskStatus],
    ) -> DevelopmentTask | None:
        """Start one task if no same-role task is already active."""
        if not from_statuses or not active_statuses:
            return None
        timestamp_text = _format_timestamp(_utc_now())
        with self._transaction("BEGIN IMMEDIATE") as connection:
            task = _select_optional_task(connection, task_id)
            if task is None or task.status not in from_statuses:
                return None
            conflict_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM development_tasks
                WHERE feature_id = ?
                    AND assigned_role = ?
                    AND id != ?
                    AND status IN (?, ?, ?, ?, ?)
                """,
                (
                    task.feature_id,
                    task.assigned_role.value,
                    task_id,
                    *_status_values(active_statuses),
                ),
            ).fetchone()
            if conflict_count is None or int(conflict_count[0]) > 0:
                return None
            connection.execute(
                """
                UPDATE development_tasks
                SET status = ?, updated_at = ?
                WHERE id = ?
                    AND status IN (?, ?, ?, ?, ?)
                """,
                (
                    TaskStatus.IN_PROGRESS.value,
                    timestamp_text,
                    task_id,
                    *_status_values(from_statuses),
                ),
            )
            return _select_optional_task(connection, task_id)

    def transition_task_status(
        self,
        task_id: int,
        from_statuses: frozenset[TaskStatus],
        to_status: TaskStatus,
    ) -> DevelopmentTask | None:
        """Compare-and-set a task status transition."""
        if not from_statuses:
            return None
        timestamp_text = _format_timestamp(_utc_now())
        with self._transaction("BEGIN IMMEDIATE") as connection:
            cursor = connection.execute(
                """
                UPDATE development_tasks
                SET status = ?, updated_at = ?
                WHERE id = ?
                    AND status IN (?, ?, ?, ?, ?)
                """,
                (
                    to_status.value,
                    timestamp_text,
                    task_id,
                    *_status_values(from_statuses),
                ),
            )
            if cursor.rowcount < 1:
                return None
            return _select_optional_task(connection, task_id)

    def submit_task_handoff(
        self,
        draft: TaskHandoffDraft,
        from_status: TaskStatus,
        to_status: TaskStatus,
    ) -> TaskHandoff | None:
        """Persist a handoff and move the task to verification."""
        timestamp_text = _format_timestamp(_utc_now())
        with self._transaction("BEGIN IMMEDIATE") as connection:
            task = _select_optional_task(connection, draft.task_id)
            if task is None or task.status is not from_status:
                return None
            cursor = connection.execute(
                """
                INSERT INTO task_handoffs (
                    task_id,
                    agent_run_id,
                    submitted_by,
                    attribution,
                    implementation_summary,
                    changed_paths_json,
                    reused_symbols_json,
                    new_symbols_json,
                    reuse_notes,
                    checks_attempted_json,
                    limitations,
                    next_action,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.task_id,
                    draft.agent_run_id,
                    draft.submitted_by.value,
                    draft.attribution,
                    draft.implementation_summary,
                    _json_text(draft.changed_paths),
                    _json_text(draft.reused_symbols),
                    _json_text(draft.new_symbols),
                    draft.reuse_notes,
                    _json_text(draft.checks_attempted),
                    draft.limitations,
                    draft.next_action,
                    timestamp_text,
                ),
            )
            connection.execute(
                """
                UPDATE development_tasks
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (to_status.value, timestamp_text, draft.task_id),
            )
            return _select_handoff(connection, _last_insert_id(cursor))

    def latest_task_handoff(self, task_id: int) -> TaskHandoff | None:
        """Return the newest persisted handoff for a task, if present."""
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    task_id,
                    agent_run_id,
                    submitted_by,
                    attribution,
                    implementation_summary,
                    changed_paths_json,
                    reused_symbols_json,
                    new_symbols_json,
                    reuse_notes,
                    checks_attempted_json,
                    limitations,
                    next_action,
                    created_at
                FROM task_handoffs
                WHERE task_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            return None if row is None else _map_handoff(row)

    def record_task_verification(
        self,
        task_id: int,
        submission_id: int,
        result: TaskVerificationResult,
        next_status: TaskStatus,
    ) -> TaskVerificationEvidence:
        """Persist verification evidence and apply the resulting status."""
        timestamp_text = _format_timestamp(_utc_now())
        with self._transaction("BEGIN IMMEDIATE") as connection:
            existing = _select_verification_by_submission(
                connection,
                submission_id,
            )
            if existing is not None:
                return existing
            cursor = connection.execute(
                """
                INSERT INTO task_verifications (
                    task_id,
                    submission_id,
                    verifier_name,
                    outcome,
                    failure_classification,
                    feedback,
                    started_at,
                    ended_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    submission_id,
                    result.verifier_name,
                    result.outcome.value,
                    result.failure_classification.value,
                    result.feedback,
                    _format_timestamp(result.started_at),
                    _format_timestamp(result.ended_at),
                ),
            )
            verification_id = _last_insert_id(cursor)
            for check in result.checks:
                connection.execute(
                    """
                    INSERT INTO task_verification_checks (
                        verification_id,
                        name,
                        started_at,
                        ended_at,
                        exit_code,
                        timed_out,
                        stdout_hash,
                        stdout_excerpt,
                        stderr_hash,
                        stderr_excerpt
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        verification_id,
                        check.name,
                        _format_timestamp(check.started_at),
                        _format_timestamp(check.ended_at),
                        check.exit_code,
                        int(check.timed_out),
                        check.stdout_hash,
                        check.stdout_excerpt,
                        check.stderr_hash,
                        check.stderr_excerpt,
                    ),
                )
            connection.execute(
                """
                UPDATE development_tasks
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (next_status.value, timestamp_text, task_id),
            )
            return _select_verification(connection, verification_id)

    def latest_task_verification(
        self,
        task_id: int,
    ) -> TaskVerificationEvidence | None:
        """Return the newest persisted verification evidence for a task."""
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    task_id,
                    submission_id,
                    verifier_name,
                    outcome,
                    failure_classification,
                    feedback,
                    started_at,
                    ended_at
                FROM task_verifications
                WHERE task_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            return None if row is None else _map_verification(connection, row)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(
        self,
        begin_statement: str = "BEGIN",
    ) -> Generator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute(begin_statement)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _select_feature(
    connection: sqlite3.Connection,
    feature_id: int,
) -> Feature:
    row = _require_row(
        connection.execute(
            """
            SELECT id, title, description, status, created_at, updated_at
            FROM features
            WHERE id = ?
            """,
            (feature_id,),
        ).fetchone(),
        "feature",
    )
    return _map_feature(row)


def _select_artifact(
    connection: sqlite3.Connection,
    artifact_id: int,
) -> Artifact:
    row = _require_row(
        connection.execute(
            """
            SELECT id, feature_id, kind, content, created_by, created_at
            FROM artifacts
            WHERE id = ?
            """,
            (artifact_id,),
        ).fetchone(),
        "artifact",
    )
    return _map_artifact(row)


def _select_optional_task(
    connection: sqlite3.Connection,
    task_id: int,
) -> DevelopmentTask | None:
    row = connection.execute(
        f"""
        {_TASK_SELECT_SQL}
        WHERE id = ?
        """,
        (task_id,),
    ).fetchone()
    return None if row is None else _map_development_task(row)


def _select_task(
    connection: sqlite3.Connection,
    task_id: int,
) -> DevelopmentTask:
    row = _require_row(
        connection.execute(
            f"""
            {_TASK_SELECT_SQL}
            WHERE id = ?
            """,
            (task_id,),
        ).fetchone(),
        "development task",
    )
    return _map_development_task(row)


def _select_handoff(
    connection: sqlite3.Connection,
    handoff_id: int,
) -> TaskHandoff:
    row = _require_row(
        connection.execute(
            """
            SELECT
                id,
                task_id,
                agent_run_id,
                submitted_by,
                attribution,
                implementation_summary,
                changed_paths_json,
                reused_symbols_json,
                new_symbols_json,
                reuse_notes,
                checks_attempted_json,
                limitations,
                next_action,
                created_at
            FROM task_handoffs
            WHERE id = ?
            """,
            (handoff_id,),
        ).fetchone(),
        "task handoff",
    )
    return _map_handoff(row)


def _select_verification(
    connection: sqlite3.Connection,
    verification_id: int,
) -> TaskVerificationEvidence:
    row = _require_row(
        connection.execute(
            """
            SELECT
                id,
                task_id,
                submission_id,
                verifier_name,
                outcome,
                failure_classification,
                feedback,
                started_at,
                ended_at
            FROM task_verifications
            WHERE id = ?
            """,
            (verification_id,),
        ).fetchone(),
        "task verification",
    )
    return _map_verification(connection, row)


def _select_verification_by_submission(
    connection: sqlite3.Connection,
    submission_id: int,
) -> TaskVerificationEvidence | None:
    row = connection.execute(
        """
        SELECT
            id,
            task_id,
            submission_id,
            verifier_name,
            outcome,
            failure_classification,
            feedback,
            started_at,
            ended_at
        FROM task_verifications
        WHERE submission_id = ?
        """,
        (submission_id,),
    ).fetchone()
    return None if row is None else _map_verification(connection, row)


def _select_verification_checks(
    connection: sqlite3.Connection,
    verification_id: int,
) -> tuple[TaskVerificationCheck, ...]:
    rows = connection.execute(
        """
        SELECT
            id,
            verification_id,
            name,
            started_at,
            ended_at,
            exit_code,
            timed_out,
            stdout_hash,
            stdout_excerpt,
            stderr_hash,
            stderr_excerpt
        FROM task_verification_checks
        WHERE verification_id = ?
        ORDER BY id
        """,
        (verification_id,),
    ).fetchall()
    return tuple(_map_verification_check(row) for row in rows)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _last_insert_id(cursor: sqlite3.Cursor) -> int:
    row_id = cursor.lastrowid
    if row_id is None:
        raise RuntimeError("SQLite did not return a new row ID.")
    return row_id


def _require_row(row: sqlite3.Row | None, record_name: str) -> sqlite3.Row:
    if row is None:
        raise RuntimeError(f"SQLite did not return the {record_name}.")
    return row


def _map_feature(row: sqlite3.Row) -> Feature:
    return Feature(
        id=int(row["id"]),
        title=str(row["title"]),
        description=str(row["description"]),
        status=FeatureStatus(str(row["status"])),
        created_at=_parse_timestamp(str(row["created_at"])),
        updated_at=_parse_timestamp(str(row["updated_at"])),
    )


def _map_artifact(row: sqlite3.Row) -> Artifact:
    return Artifact(
        id=int(row["id"]),
        feature_id=int(row["feature_id"]),
        kind=ArtifactKind(str(row["kind"])),
        content=str(row["content"]),
        created_by=str(row["created_by"]),
        created_at=_parse_timestamp(str(row["created_at"])),
    )


def _map_development_task(row: sqlite3.Row) -> DevelopmentTask:
    role = DevelopmentRole(str(row["assigned_role"]))
    return DevelopmentTask(
        id=int(row["id"]),
        feature_id=int(row["feature_id"]),
        title=str(row["title"]),
        description=str(row["description"]),
        assigned_role=role,
        status=TaskStatus(str(row["status"])),
        created_at=_parse_timestamp(str(row["created_at"])),
        updated_at=_parse_timestamp(str(row["updated_at"])),
        verification_contract=_map_verification_contract(row, role),
    )


def _map_handoff(row: sqlite3.Row) -> TaskHandoff:
    return TaskHandoff(
        id=int(row["id"]),
        task_id=int(row["task_id"]),
        agent_run_id=int(row["agent_run_id"]),
        submitted_by=DevelopmentRole(str(row["submitted_by"])),
        attribution=str(row["attribution"]),
        implementation_summary=str(row["implementation_summary"]),
        changed_paths=_json_tuple(str(row["changed_paths_json"])),
        reused_symbols=_json_tuple(str(row["reused_symbols_json"])),
        new_symbols=_json_tuple(str(row["new_symbols_json"])),
        reuse_notes=str(row["reuse_notes"]),
        checks_attempted=_json_tuple(str(row["checks_attempted_json"])),
        limitations=str(row["limitations"]),
        next_action=str(row["next_action"]),
        created_at=_parse_timestamp(str(row["created_at"])),
    )


def _map_verification(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> TaskVerificationEvidence:
    verification_id = int(row["id"])
    return TaskVerificationEvidence(
        id=verification_id,
        task_id=int(row["task_id"]),
        submission_id=int(row["submission_id"]),
        verifier_name=str(row["verifier_name"]),
        outcome=TaskVerificationOutcome(str(row["outcome"])),
        failure_classification=FailureClassification(
            str(row["failure_classification"]),
        ),
        feedback=str(row["feedback"]),
        checks=_select_verification_checks(connection, verification_id),
        started_at=_parse_timestamp(str(row["started_at"])),
        ended_at=_parse_timestamp(str(row["ended_at"])),
    )


def _map_verification_check(row: sqlite3.Row) -> TaskVerificationCheck:
    return TaskVerificationCheck(
        id=int(row["id"]),
        verification_id=int(row["verification_id"]),
        name=str(row["name"]),
        started_at=_parse_timestamp(str(row["started_at"])),
        ended_at=_parse_timestamp(str(row["ended_at"])),
        exit_code=int(row["exit_code"]),
        timed_out=bool(row["timed_out"]),
        stdout_hash=str(row["stdout_hash"]),
        stdout_excerpt=str(row["stdout_excerpt"]),
        stderr_hash=str(row["stderr_hash"]),
        stderr_excerpt=str(row["stderr_excerpt"]),
    )


def _map_verification_contract(
    row: sqlite3.Row,
    assigned_role: DevelopmentRole,
) -> TaskVerificationContract | None:
    profile_value = row["verification_profile"]
    checks_value = row["required_checks_json"]
    if profile_value is None or checks_value is None:
        return default_verification_contract(assigned_role)
    return TaskVerificationContract(
        profile_name=str(profile_value),
        required_checks=_json_tuple(str(checks_value)),
    )


def _contract_checks_json(
    contract: TaskVerificationContract | None,
) -> str | None:
    if contract is None:
        return None
    return _json_text(contract.required_checks)


def _json_text(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _json_tuple(value: str) -> tuple[str, ...]:
    parsed: object = json.loads(value)
    if not isinstance(parsed, list):
        raise RuntimeError("SQLite stored invalid workflow JSON.")
    parsed_items = cast("list[object]", parsed)
    if not all(isinstance(item, str) for item in parsed_items):
        raise RuntimeError("SQLite stored invalid workflow JSON.")
    items = cast("list[str]", parsed_items)
    return tuple(items)


def _status_values(statuses: frozenset[TaskStatus]) -> tuple[str, ...]:
    return tuple(
        status.value if status in statuses else _UNMATCHED_STATUS
        for status in TaskStatus
    )
