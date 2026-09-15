"""SQLite workflow schema migrations."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent_team.infrastructure.persistence.sqlite.workflow import (
    sqlite_workflow_migration_error,
)

_CREATE_FEATURES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_CREATE_ARTIFACTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE
);
"""

_CREATE_DEVELOPMENT_TASKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS development_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    assigned_role TEXT NOT NULL,
    status TEXT NOT NULL,
    verification_profile TEXT,
    required_checks_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE
);
"""

_CREATE_TASK_HANDOFFS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_handoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    agent_run_id INTEGER NOT NULL,
    submitted_by TEXT NOT NULL,
    attribution TEXT NOT NULL,
    implementation_summary TEXT NOT NULL,
    changed_paths_json TEXT NOT NULL,
    reused_symbols_json TEXT NOT NULL,
    new_symbols_json TEXT NOT NULL,
    reuse_notes TEXT NOT NULL,
    checks_attempted_json TEXT NOT NULL,
    limitations TEXT NOT NULL,
    next_action TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES development_tasks(id) ON DELETE CASCADE
);
"""

_CREATE_TASK_VERIFICATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    submission_id INTEGER NOT NULL UNIQUE,
    verifier_name TEXT NOT NULL,
    outcome TEXT NOT NULL,
    failure_classification TEXT NOT NULL,
    feedback TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES development_tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (submission_id) REFERENCES task_handoffs(id) ON DELETE CASCADE
);
"""

_CREATE_TASK_VERIFICATION_CHECKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_verification_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    verification_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    timed_out INTEGER NOT NULL,
    stdout_hash TEXT NOT NULL,
    stdout_excerpt TEXT NOT NULL,
    stderr_hash TEXT NOT NULL,
    stderr_excerpt TEXT NOT NULL,
    FOREIGN KEY (verification_id)
        REFERENCES task_verifications(id)
        ON DELETE CASCADE
);
"""

_DEVELOPMENT_TASK_CONTRACT_COLUMNS = (
    ("verification_profile", "TEXT"),
    ("required_checks_json", "TEXT"),
)

_INDEXES_SQL = (
    """
    CREATE INDEX IF NOT EXISTS idx_artifacts_feature_id
    ON artifacts(feature_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_development_tasks_feature_id
    ON development_tasks(feature_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_development_tasks_active
    ON development_tasks(feature_id, assigned_role, status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_task_handoffs_task_id
    ON task_handoffs(task_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_task_handoffs_run_id
    ON task_handoffs(agent_run_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_task_verifications_task_id
    ON task_verifications(task_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_task_verification_checks_verification_id
    ON task_verification_checks(verification_id)
    """,
)


@dataclass(frozen=True, slots=True)
class SQLiteWorkflowSchemaMigrator:
    """Apply idempotent workflow schema migrations atomically."""

    database_path: Path

    def migrate(self) -> None:
        """Create or migrate workflow tables inside one transaction."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN")
            self._ensure_base_tables(connection)
            self._ensure_task_contract_columns(connection)
            self._create_indexes(connection)
            connection.commit()
        except sqlite3.Error as error:
            connection.rollback()
            raise sqlite_workflow_migration_error.SQLiteWorkflowMigrationError(
                "Workflow database migration failed.",
            ) from error
        finally:
            connection.close()

    def _ensure_base_tables(self, connection: sqlite3.Connection) -> None:
        """Create workflow tables if they do not already exist."""
        connection.execute(_CREATE_FEATURES_TABLE_SQL)
        connection.execute(_CREATE_ARTIFACTS_TABLE_SQL)
        connection.execute(_CREATE_DEVELOPMENT_TASKS_TABLE_SQL)
        connection.execute(_CREATE_TASK_HANDOFFS_TABLE_SQL)
        connection.execute(_CREATE_TASK_VERIFICATIONS_TABLE_SQL)
        connection.execute(_CREATE_TASK_VERIFICATION_CHECKS_TABLE_SQL)

    def _ensure_task_contract_columns(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """Add nullable task verification contract columns."""
        columns = _table_columns(connection, "development_tasks")
        for column_name, column_type in _DEVELOPMENT_TASK_CONTRACT_COLUMNS:
            if column_name not in columns:
                connection.execute(
                    "ALTER TABLE development_tasks "
                    f"ADD COLUMN {column_name} {column_type}",
                )

    def _create_indexes(self, connection: sqlite3.Connection) -> None:
        """Create workflow indexes after tables and columns exist."""
        for index_sql in _INDEXES_SQL:
            connection.execute(index_sql)


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}
