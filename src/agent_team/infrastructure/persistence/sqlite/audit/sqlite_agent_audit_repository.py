"""SQLite-backed agent audit repository."""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from agent_team.application.audit.audit_sanitizer import (
    generation_metadata_to_json,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.audit.agent_run_status import AgentRunStatus
from agent_team.domain.audit.tool_classification import ToolClassification
from agent_team.domain.audit.tool_invocation_denial import ToolInvocationDenial
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.domain.audit.tool_invocation_start import ToolInvocationStart
from agent_team.domain.audit.tool_invocation_status import ToolInvocationStatus
from agent_team.domain.runtime.agent_generation_metadata import (
    AgentGenerationMetadata,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_audit_schema_migrator as audit_schema_migrator,
)


@dataclass(frozen=True, slots=True)
class SQLiteAgentAuditRepository:
    """SQLite implementation of local agent auditing."""

    database_path: Path

    def __post_init__(self) -> None:
        """Create the database directory and audit schema."""
        audit_schema_migrator.SQLiteAuditSchemaMigrator(
            self.database_path,
        ).migrate()

    def start_run(
        self,
        run: AgentRunStart,
    ) -> AgentRunRecord:
        """Record the start of an agent run."""
        started_at = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_runs (
                    role,
                    model,
                    status,
                    prompt_hash,
                    prompt_excerpt,
                    started_at,
                    max_turns,
                    session_id,
                    feature_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.role.value,
                    run.model,
                    AgentRunStatus.STARTED.value,
                    run.prompt_hash,
                    run.prompt_excerpt,
                    started_at,
                    run.max_turns,
                    run.session_id,
                    run.feature_id,
                ),
            )
            return _select_run(connection, _last_insert_id(cursor))

    def complete_run(
        self,
        run_id: int,
        output_hash: str,
        output_excerpt: str,
        generation_metadata: AgentGenerationMetadata | None = None,
    ) -> AgentRunRecord:
        """Finalize an agent run as completed."""
        ended_at = _format_timestamp(_utc_now())
        generation_metadata_json = generation_metadata_to_json(
            generation_metadata,
        )
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET
                    status = ?,
                    ended_at = ?,
                    output_hash = ?,
                    output_excerpt = ?,
                    generation_metadata_json = ?,
                    error_type = NULL,
                    error_message = NULL
                WHERE id = ?
                """,
                (
                    AgentRunStatus.COMPLETED.value,
                    ended_at,
                    output_hash,
                    output_excerpt,
                    generation_metadata_json,
                    run_id,
                ),
            )
            return _select_run(connection, run_id)

    def fail_run(
        self,
        run_id: int,
        error_type: str,
        error_message: str,
    ) -> AgentRunRecord:
        """Finalize an agent run as failed."""
        ended_at = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET
                    status = ?,
                    ended_at = ?,
                    error_type = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    AgentRunStatus.FAILED.value,
                    ended_at,
                    error_type,
                    error_message,
                    run_id,
                ),
            )
            return _select_run(connection, run_id)

    def record_run_generation_metadata(
        self,
        run_id: int,
        output_hash: str,
        output_excerpt: str,
        generation_metadata: AgentGenerationMetadata,
    ) -> AgentRunRecord:
        """Record sanitized model-generation metadata for an agent run."""
        generation_metadata_json = generation_metadata_to_json(
            generation_metadata,
        )
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET
                    output_hash = ?,
                    output_excerpt = ?,
                    generation_metadata_json = ?
                WHERE id = ?
                """,
                (
                    output_hash,
                    output_excerpt,
                    generation_metadata_json,
                    run_id,
                ),
            )
            return _select_run(connection, run_id)

    def start_tool_invocation(
        self,
        invocation: ToolInvocationStart,
    ) -> ToolInvocationRecord:
        """Record an allowed tool invocation before execution."""
        started_at = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tool_invocations (
                    run_id,
                    server_name,
                    tool_name,
                    classification,
                    status,
                    arguments_hash,
                    arguments_preview_json,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invocation.run_id,
                    invocation.server_name,
                    invocation.tool_name,
                    invocation.classification.value,
                    ToolInvocationStatus.ALLOWED.value,
                    invocation.arguments_hash,
                    invocation.arguments_preview_json,
                    started_at,
                ),
            )
            return _select_tool_invocation(
                connection,
                _last_insert_id(cursor),
            )

    def complete_tool_invocation(
        self,
        invocation_id: int,
        result_hash: str,
        result_preview: str,
    ) -> ToolInvocationRecord:
        """Finalize a tool invocation as completed."""
        ended_at = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE tool_invocations
                SET
                    status = ?,
                    ended_at = ?,
                    result_hash = ?,
                    result_preview = ?,
                    error_type = NULL,
                    error_message = NULL
                WHERE id = ?
                """,
                (
                    ToolInvocationStatus.COMPLETED.value,
                    ended_at,
                    result_hash,
                    result_preview,
                    invocation_id,
                ),
            )
            return _select_tool_invocation(connection, invocation_id)

    def fail_tool_invocation(
        self,
        invocation_id: int,
        error_type: str,
        error_message: str,
    ) -> ToolInvocationRecord:
        """Finalize a tool invocation as failed."""
        ended_at = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE tool_invocations
                SET
                    status = ?,
                    ended_at = ?,
                    error_type = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    ToolInvocationStatus.FAILED.value,
                    ended_at,
                    error_type,
                    error_message,
                    invocation_id,
                ),
            )
            return _select_tool_invocation(connection, invocation_id)

    def deny_tool_invocation(
        self,
        denial: ToolInvocationDenial,
    ) -> ToolInvocationRecord:
        """Record a denied tool invocation without execution."""
        timestamp = _format_timestamp(_utc_now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tool_invocations (
                    run_id,
                    server_name,
                    tool_name,
                    classification,
                    status,
                    arguments_hash,
                    arguments_preview_json,
                    started_at,
                    ended_at,
                    error_type,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    denial.invocation.run_id,
                    denial.invocation.server_name,
                    denial.invocation.tool_name,
                    denial.invocation.classification.value,
                    ToolInvocationStatus.DENIED.value,
                    denial.invocation.arguments_hash,
                    denial.invocation.arguments_preview_json,
                    timestamp,
                    timestamp,
                    denial.error_type,
                    denial.error_message,
                ),
            )
            return _select_tool_invocation(
                connection,
                _last_insert_id(cursor),
            )

    def list_runs(self, limit: int) -> list[AgentRunRecord]:
        """Return recent agent runs up to the requested limit."""
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    role,
                    model,
                    status,
                    prompt_hash,
                    prompt_excerpt,
                    started_at,
                    ended_at,
                    max_turns,
                    output_hash,
                    output_excerpt,
                    error_type,
                    error_message,
                    session_id,
                    feature_id,
                    generation_metadata_json
                FROM agent_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [_map_run(row) for row in rows]

    def get_run(self, run_id: int) -> AgentRunRecord | None:
        """Return one agent run by ID, if it exists."""
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    role,
                    model,
                    status,
                    prompt_hash,
                    prompt_excerpt,
                    started_at,
                    ended_at,
                    max_turns,
                    output_hash,
                    output_excerpt,
                    error_type,
                    error_message,
                    session_id,
                    feature_id,
                    generation_metadata_json
                FROM agent_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            return None if row is None else _map_run(row)

    def list_tool_invocations(
        self,
        run_id: int,
    ) -> list[ToolInvocationRecord]:
        """Return tool invocations for one agent run."""
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    run_id,
                    server_name,
                    tool_name,
                    classification,
                    status,
                    arguments_hash,
                    arguments_preview_json,
                    result_hash,
                    result_preview,
                    started_at,
                    ended_at,
                    error_type,
                    error_message
                FROM tool_invocations
                WHERE run_id = ?
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
            return [_map_tool_invocation(row) for row in rows]

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


def _select_run(
    connection: sqlite3.Connection,
    run_id: int,
) -> AgentRunRecord:
    row = _require_row(
        connection.execute(
            """
            SELECT
                id,
                role,
                model,
                status,
                prompt_hash,
                prompt_excerpt,
                started_at,
                ended_at,
                max_turns,
                output_hash,
                output_excerpt,
                error_type,
                error_message,
                session_id,
                feature_id,
                generation_metadata_json
            FROM agent_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone(),
        "agent run",
    )
    return _map_run(row)


def _select_tool_invocation(
    connection: sqlite3.Connection,
    invocation_id: int,
) -> ToolInvocationRecord:
    row = _require_row(
        connection.execute(
            """
            SELECT
                id,
                run_id,
                server_name,
                tool_name,
                classification,
                status,
                arguments_hash,
                arguments_preview_json,
                result_hash,
                result_preview,
                started_at,
                ended_at,
                error_type,
                error_message
            FROM tool_invocations
            WHERE id = ?
            """,
            (invocation_id,),
        ).fetchone(),
        "tool invocation",
    )
    return _map_tool_invocation(row)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
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


def _map_run(row: sqlite3.Row) -> AgentRunRecord:
    return AgentRunRecord(
        id=int(row["id"]),
        role=DevelopmentRole(str(row["role"])),
        model=str(row["model"]),
        status=AgentRunStatus(str(row["status"])),
        prompt_hash=str(row["prompt_hash"]),
        prompt_excerpt=str(row["prompt_excerpt"]),
        started_at=_require_timestamp(row["started_at"]),
        ended_at=_parse_timestamp(row["ended_at"]),
        max_turns=int(row["max_turns"]),
        output_hash=_optional_text(row["output_hash"]),
        output_excerpt=_optional_text(row["output_excerpt"]),
        error_type=_optional_text(row["error_type"]),
        error_message=_optional_text(row["error_message"]),
        session_id=_optional_text(row["session_id"]),
        feature_id=_optional_int(row["feature_id"]),
        generation_metadata=_generation_metadata(
            row["generation_metadata_json"],
        ),
    )


def _map_tool_invocation(row: sqlite3.Row) -> ToolInvocationRecord:
    return ToolInvocationRecord(
        id=int(row["id"]),
        run_id=int(row["run_id"]),
        server_name=str(row["server_name"]),
        tool_name=str(row["tool_name"]),
        classification=ToolClassification(str(row["classification"])),
        status=ToolInvocationStatus(str(row["status"])),
        arguments_hash=str(row["arguments_hash"]),
        arguments_preview_json=str(row["arguments_preview_json"]),
        result_hash=_optional_text(row["result_hash"]),
        result_preview=_optional_text(row["result_preview"]),
        started_at=_require_timestamp(row["started_at"]),
        ended_at=_parse_timestamp(row["ended_at"]),
        error_type=_optional_text(row["error_type"]),
        error_message=_optional_text(row["error_message"]),
    )


def _require_timestamp(value: object) -> datetime:
    timestamp = _parse_timestamp(str(value))
    if timestamp is None:
        raise RuntimeError("SQLite returned a missing timestamp.")
    return timestamp


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(str(value))


def _generation_metadata(value: object) -> AgentGenerationMetadata | None:
    text = _optional_text(value)
    if text is None:
        return None
    parsed: object = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("SQLite returned invalid generation metadata.")
    data = cast("dict[str, object]", parsed)
    return AgentGenerationMetadata(
        finish_reason=_optional_metadata_text(data.get("finish_reason")),
        input_tokens=_optional_metadata_int(data.get("input_tokens")),
        output_tokens=_optional_metadata_int(data.get("output_tokens")),
        visible_output_char_count=_metadata_int(
            data.get("visible_output_char_count"),
        ),
        objectively_truncated=_metadata_bool(
            data.get("objectively_truncated"),
        ),
        model=_metadata_text(data.get("model")),
    )


def _optional_metadata_text(value: object) -> str | None:
    if value is None:
        return None
    return _metadata_text(value)


def _metadata_text(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("SQLite returned invalid generation metadata.")
    return value


def _optional_metadata_int(value: object) -> int | None:
    if value is None:
        return None
    return _metadata_int(value)


def _metadata_int(value: object) -> int:
    if not isinstance(value, int):
        raise RuntimeError("SQLite returned invalid generation metadata.")
    return value


def _metadata_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError("SQLite returned invalid generation metadata.")
    return value
