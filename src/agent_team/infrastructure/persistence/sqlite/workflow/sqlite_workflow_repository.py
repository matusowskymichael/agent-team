"""SQLite-backed workflow repository."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact import Artifact
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.development_task import DevelopmentTask
from agent_team.domain.workflow.feature import Feature
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS development_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    assigned_role TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_artifacts_feature_id
ON artifacts(feature_id);

CREATE INDEX IF NOT EXISTS idx_development_tasks_feature_id
ON development_tasks(feature_id);
"""


@dataclass(frozen=True, slots=True)
class SQLiteWorkflowRepository:
    """SQLite implementation of workflow persistence."""

    database_path: Path

    def __post_init__(self) -> None:
        """Create the database directory and schema."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.executescript(SCHEMA_SQL)

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
            feature_id = _last_insert_id(cursor)
            row = _require_row(
                connection.execute(
                    """
                    SELECT
                        id,
                        title,
                        description,
                        status,
                        created_at,
                        updated_at
                    FROM features
                    WHERE id = ?
                    """,
                    (feature_id,),
                ).fetchone(),
                "created feature",
            )
            return _map_feature(row)

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
            artifact_id = _last_insert_id(cursor)
            row = _require_row(
                connection.execute(
                    """
                    SELECT
                        id,
                        feature_id,
                        kind,
                        content,
                        created_by,
                        created_at
                    FROM artifacts
                    WHERE id = ?
                    """,
                    (artifact_id,),
                ).fetchone(),
                "created artifact",
            )
            return _map_artifact(row)

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
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO development_tasks (
                    feature_id,
                    title,
                    description,
                    assigned_role,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feature_id,
                    title,
                    description,
                    assigned_role.value,
                    status.value,
                    timestamp_text,
                    timestamp_text,
                ),
            )
            task_id = _last_insert_id(cursor)
            row = _require_row(
                connection.execute(
                    """
                    SELECT
                        id,
                        feature_id,
                        title,
                        description,
                        assigned_role,
                        status,
                        created_at,
                        updated_at
                    FROM development_tasks
                    WHERE id = ?
                    """,
                    (task_id,),
                ).fetchone(),
                "created development task",
            )
            return _map_development_task(row)

    def get_task(self, task_id: int) -> DevelopmentTask | None:
        """Return a development task by ID, if it exists."""
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    feature_id,
                    title,
                    description,
                    assigned_role,
                    status,
                    created_at,
                    updated_at
                FROM development_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
            return None if row is None else _map_development_task(row)

    def list_tasks(self, feature_id: int) -> list[DevelopmentTask]:
        """Return development tasks attached to a feature."""
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    feature_id,
                    title,
                    description,
                    assigned_role,
                    status,
                    created_at,
                    updated_at
                FROM development_tasks
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
            row = connection.execute(
                """
                SELECT
                    id,
                    feature_id,
                    title,
                    description,
                    assigned_role,
                    status,
                    created_at,
                    updated_at
                FROM development_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
            return None if row is None else _map_development_task(row)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


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
    return DevelopmentTask(
        id=int(row["id"]),
        feature_id=int(row["feature_id"]),
        title=str(row["title"]),
        description=str(row["description"]),
        assigned_role=DevelopmentRole(str(row["assigned_role"])),
        status=TaskStatus(str(row["status"])),
        created_at=_parse_timestamp(str(row["created_at"])),
        updated_at=_parse_timestamp(str(row["updated_at"])),
    )
