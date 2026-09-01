"""SQLite audit schema migrations."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_audit_migration_error,
)

CURRENT_AUDIT_SCHEMA_VERSION = 3
AUDIT_SCHEMA_SESSION_METADATA_VERSION = 2
AUDIT_SCHEMA_GENERATION_METADATA_VERSION = 3

_CREATE_AGENT_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_excerpt TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    max_turns INTEGER NOT NULL,
    output_hash TEXT,
    output_excerpt TEXT,
    error_type TEXT,
    error_message TEXT,
    generation_metadata_json TEXT
);
"""

_CREATE_TOOL_INVOCATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tool_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    server_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    classification TEXT NOT NULL,
    status TEXT NOT NULL,
    arguments_hash TEXT NOT NULL,
    arguments_preview_json TEXT NOT NULL,
    result_hash TEXT,
    result_preview TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    error_type TEXT,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
);
"""

_AGENT_RUN_METADATA_COLUMNS = (
    ("session_id", "TEXT"),
    ("feature_id", "INTEGER"),
)

_AGENT_RUN_GENERATION_METADATA_COLUMNS = (
    ("generation_metadata_json", "TEXT"),
)

_INDEXES_SQL = (
    """
    CREATE INDEX IF NOT EXISTS idx_agent_runs_role
    ON agent_runs(role)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_runs_status
    ON agent_runs(status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at
    ON agent_runs(started_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_runs_session_id
    ON agent_runs(session_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_runs_feature_id
    ON agent_runs(feature_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_invocations_run_id
    ON tool_invocations(run_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_invocations_tool_name
    ON tool_invocations(tool_name)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_invocations_status
    ON tool_invocations(status)
    """,
)


@dataclass(frozen=True, slots=True)
class SQLiteAuditSchemaMigrator:
    """Apply ordered audit schema migrations to a SQLite database."""

    database_path: Path

    def migrate(self) -> None:
        """Create or migrate the audit schema inside one transaction."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN")
            version = _schema_version(connection)
            if version < 1:
                self._migrate_to_version_1(connection)
            else:
                self._ensure_base_tables(connection)
            if version < AUDIT_SCHEMA_SESSION_METADATA_VERSION:
                self._migrate_to_version_2(connection)
            else:
                self._ensure_session_columns(connection)
            if version < AUDIT_SCHEMA_GENERATION_METADATA_VERSION:
                self._migrate_to_version_3(connection)
            else:
                self._ensure_generation_columns(connection)
            self._create_indexes(connection)
            connection.execute(
                f"PRAGMA user_version = {CURRENT_AUDIT_SCHEMA_VERSION}",
            )
            connection.commit()
        except sqlite3.Error as error:
            connection.rollback()
            raise sqlite_audit_migration_error.SQLiteAuditMigrationError(
                "Audit database migration failed.",
            ) from error
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate_to_version_1(self, connection: sqlite3.Connection) -> None:
        """Create the original audit tables."""
        self._ensure_base_tables(connection)

    def _migrate_to_version_2(self, connection: sqlite3.Connection) -> None:
        """Add feature-scoped run metadata columns."""
        self._ensure_session_columns(connection)

    def _migrate_to_version_3(self, connection: sqlite3.Connection) -> None:
        """Add sanitized model-generation metadata."""
        self._ensure_generation_columns(connection)

    def _ensure_base_tables(self, connection: sqlite3.Connection) -> None:
        """Create the base audit tables if they do not already exist."""
        connection.execute(_CREATE_AGENT_RUNS_TABLE_SQL)
        connection.execute(_CREATE_TOOL_INVOCATIONS_TABLE_SQL)

    def _ensure_session_columns(self, connection: sqlite3.Connection) -> None:
        """Add nullable columns for feature-scoped sessions."""
        for column_name, column_type in _AGENT_RUN_METADATA_COLUMNS:
            _ensure_agent_run_column(connection, column_name, column_type)

    def _ensure_generation_columns(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """Add nullable columns for model generation metadata."""
        for column_name, column_type in _AGENT_RUN_GENERATION_METADATA_COLUMNS:
            _ensure_agent_run_column(connection, column_name, column_type)

    def _create_indexes(self, connection: sqlite3.Connection) -> None:
        """Create audit indexes after all referenced columns exist."""
        for index_sql in _INDEXES_SQL:
            connection.execute(index_sql)


def _schema_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _ensure_agent_run_column(
    connection: sqlite3.Connection,
    column_name: str,
    column_type: str,
) -> None:
    columns = _table_columns(connection, "agent_runs")
    if column_name not in columns:
        connection.execute(
            f"ALTER TABLE agent_runs ADD COLUMN {column_name} {column_type}",
        )


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}
