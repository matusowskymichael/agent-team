"""Integration tests for SQLite audit schema migrations."""

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest import CaptureFixture

from agent_team.domain.audit.agent_run_start import AgentRunStart
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.configuration.workflow_database_path import (
    AGENT_TEAM_DB_PATH_ENV,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_audit_migration_error,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_audit_schema_migrator as migrator_module,
)
from agent_team.interfaces.cli import audit_cli
from tests.reporting.allure_steps import report_step


class TestSQLiteAuditSchemaMigration:
    """SQLite audit migration behavior tests."""

    def test_legacy_audit_schema_migrates_without_losing_records(
        self,
        legacy_audit_database: Path,
        sqlite_connection: Callable[[Path], sqlite3.Connection],
        monkeypatch: pytest.MonkeyPatch,
        capsys: CaptureFixture[str],
    ) -> None:
        """Preserve legacy audit rows while adding current columns."""
        database_path = legacy_audit_database

        with report_step("Migrate the legacy audit schema transactionally"):
            repository = audit_repository_module.SQLiteAgentAuditRepository(
                database_path,
            )
            new_run = repository.start_run(
                AgentRunStart(
                    role=DevelopmentRole.DELIVERY_MANAGER,
                    model="qwen3.5:9b",
                    prompt_hash="new-prompt-hash",
                    prompt_excerpt="Summarize feature.",
                    max_turns=6,
                    session_id="session-new",
                    feature_id=42,
                ),
            )
            completed_run = repository.complete_run(
                run_id=new_run.id,
                output_hash="new-output-hash",
                output_excerpt="Summarized feature.",
            )

        with report_step("Verify schema, indexes, and historical records"):
            _assert_current_schema(database_path, sqlite_connection)
            _assert_legacy_rows_were_preserved(
                database_path,
                sqlite_connection,
            )
            assert _user_version(database_path, sqlite_connection) == (
                migrator_module.CURRENT_AUDIT_SCHEMA_VERSION
            )

        monkeypatch.setenv(AGENT_TEAM_DB_PATH_ENV, str(database_path))
        assert audit_cli.main(["show-run", "1"]) == 0
        old_output = capsys.readouterr()
        assert "Run 1" in old_output.out
        assert "Feature ID: -" in old_output.out
        assert "Session ID: -" in old_output.out
        assert "development_workflow.list_features" in old_output.out
        assert old_output.err == ""

        assert audit_cli.main(["show-run", str(new_run.id)]) == 0
        new_output = capsys.readouterr()
        assert "Feature ID: 42" in new_output.out
        assert "Session ID: session-new" in new_output.out
        assert "Generation metadata: -" in new_output.out
        assert new_output.err == ""
        assert completed_run.generation_metadata is None

        with report_step("Re-run migration and verify idempotence"):
            before = _table_counts(database_path, sqlite_connection)
            audit_repository_module.SQLiteAgentAuditRepository(database_path)
            assert _table_counts(database_path, sqlite_connection) == before
            assert _user_version(database_path, sqlite_connection) == (
                migrator_module.CURRENT_AUDIT_SCHEMA_VERSION
            )

    def test_fresh_database_receives_current_audit_schema(
        self,
        tmp_path: Path,
        sqlite_connection: Callable[[Path], sqlite3.Connection],
    ) -> None:
        """Create the complete current audit schema for new databases."""
        database_path = tmp_path / "workflow.db"

        audit_repository_module.SQLiteAgentAuditRepository(database_path)

        _assert_current_schema(database_path, sqlite_connection)
        assert _user_version(database_path, sqlite_connection) == (
            migrator_module.CURRENT_AUDIT_SCHEMA_VERSION
        )

    def test_migration_failure_rolls_back_schema_changes(
        self,
        legacy_audit_database: Path,
        sqlite_connection: Callable[[Path], sqlite3.Connection],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Roll back added columns when a later migration step fails."""
        database_path = legacy_audit_database

        def fail_index_creation(
            _migrator: migrator_module.SQLiteAuditSchemaMigrator,
            _connection: sqlite3.Connection,
        ) -> None:
            raise sqlite3.OperationalError("index failure")

        monkeypatch.setattr(
            migrator_module.SQLiteAuditSchemaMigrator,
            "_create_indexes",
            fail_index_creation,
        )

        with pytest.raises(
            sqlite_audit_migration_error.SQLiteAuditMigrationError,
        ) as migration_error:
            migrator_module.SQLiteAuditSchemaMigrator(database_path).migrate()

        assert isinstance(migration_error.value.__cause__, sqlite3.Error)
        assert "session_id" not in _columns(
            database_path,
            "agent_runs",
            sqlite_connection,
        )
        assert "feature_id" not in _columns(
            database_path,
            "agent_runs",
            sqlite_connection,
        )
        assert "generation_metadata_json" not in _columns(
            database_path,
            "agent_runs",
            sqlite_connection,
        )
        assert "idx_agent_runs_session_id" not in _index_names(
            database_path,
            "agent_runs",
            sqlite_connection,
        )
        assert _user_version(database_path, sqlite_connection) == 0

    def test_indexes_are_created_after_required_columns_exist(
        self,
        legacy_audit_database: Path,
        sqlite_connection: Callable[[Path], sqlite3.Connection],
    ) -> None:
        """Ensure indexes are created only after column migrations."""
        database_path = legacy_audit_database
        calls: list[str] = []

        class ObservedMigrator(
            migrator_module.SQLiteAuditSchemaMigrator,
        ):
            """Migrator that records schema state before indexing."""

            def _create_indexes(
                self,
                connection: sqlite3.Connection,
            ) -> None:
                columns = _connection_columns(connection, "agent_runs")
                assert {
                    "session_id",
                    "feature_id",
                    "generation_metadata_json",
                }.issubset(columns)
                calls.append("create_indexes")
                super()._create_indexes(connection)

        ObservedMigrator(database_path).migrate()

        assert calls == ["create_indexes"]
        assert "idx_agent_runs_session_id" in _index_names(
            database_path,
            "agent_runs",
            sqlite_connection,
        )


def _assert_current_schema(
    database_path: Path,
    connect: Callable[[Path], sqlite3.Connection],
) -> None:
    agent_run_columns = _columns(database_path, "agent_runs", connect)
    tool_invocation_columns = _columns(
        database_path,
        "tool_invocations",
        connect,
    )
    assert {
        "id",
        "role",
        "model",
        "status",
        "prompt_hash",
        "prompt_excerpt",
        "started_at",
        "ended_at",
        "max_turns",
        "output_hash",
        "output_excerpt",
        "error_type",
        "error_message",
        "session_id",
        "feature_id",
        "generation_metadata_json",
    }.issubset(agent_run_columns)
    assert {
        "id",
        "run_id",
        "server_name",
        "tool_name",
        "classification",
        "status",
        "arguments_hash",
        "arguments_preview_json",
        "result_hash",
        "result_preview",
        "started_at",
        "ended_at",
        "error_type",
        "error_message",
    }.issubset(tool_invocation_columns)
    assert {
        "idx_agent_runs_role",
        "idx_agent_runs_status",
        "idx_agent_runs_started_at",
        "idx_agent_runs_session_id",
        "idx_agent_runs_feature_id",
    }.issubset(_index_names(database_path, "agent_runs", connect))
    assert {
        "idx_tool_invocations_run_id",
        "idx_tool_invocations_tool_name",
        "idx_tool_invocations_status",
    }.issubset(_index_names(database_path, "tool_invocations", connect))


def _assert_legacy_rows_were_preserved(
    database_path: Path,
    connect: Callable[[Path], sqlite3.Connection],
) -> None:
    connection = connect(database_path)
    run = connection.execute(
        """
        SELECT
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
        WHERE id = 1
        """,
    ).fetchone()
    invocation = connection.execute(
        """
        SELECT
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
        WHERE id = 1
        """,
    ).fetchone()

    assert run == (
        "business_analyst",
        "qwen3.5:9b",
        "completed",
        "legacy-prompt-hash",
        "List features.",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:02+00:00",
        6,
        "legacy-output-hash",
        "Feature list returned.",
        None,
        None,
        None,
        None,
        None,
    )
    assert invocation == (
        1,
        "development_workflow",
        "list_features",
        "read_only",
        "completed",
        "legacy-arguments-hash",
        "{}",
        "legacy-result-hash",
        "[]",
        "2026-01-01T00:00:01+00:00",
        "2026-01-01T00:00:02+00:00",
        None,
        None,
    )


def _columns(
    database_path: Path,
    table_name: str,
    connect: Callable[[Path], sqlite3.Connection],
) -> set[str]:
    return _connection_columns(connect(database_path), table_name)


def _connection_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _index_names(
    database_path: Path,
    table_name: str,
    connect: Callable[[Path], sqlite3.Connection],
) -> set[str]:
    rows = (
        connect(database_path)
        .execute(
            f"PRAGMA index_list({table_name})",
        )
        .fetchall()
    )
    return {str(row[1]) for row in rows}


def _table_counts(
    database_path: Path,
    connect: Callable[[Path], sqlite3.Connection],
) -> tuple[int, int]:
    connection = connect(database_path)
    agent_runs = connection.execute(
        "SELECT COUNT(*) FROM agent_runs",
    ).fetchone()[0]
    tool_invocations = connection.execute(
        "SELECT COUNT(*) FROM tool_invocations",
    ).fetchone()[0]
    return int(agent_runs), int(tool_invocations)


def _user_version(
    database_path: Path,
    connect: Callable[[Path], sqlite3.Connection],
) -> int:
    connection = connect(database_path)
    return int(connection.execute("PRAGMA user_version").fetchone()[0])
