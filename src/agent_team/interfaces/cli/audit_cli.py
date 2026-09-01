"""Human-only command-line interface for audit inspection."""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime

from agent_team.application.audit.audit_query_service import AuditQueryService
from agent_team.domain.audit.agent_run_details import AgentRunDetails
from agent_team.domain.audit.agent_run_not_found_error import (
    AgentRunNotFoundError,
)
from agent_team.domain.audit.agent_run_record import AgentRunRecord
from agent_team.domain.audit.tool_invocation_record import ToolInvocationRecord
from agent_team.infrastructure.configuration.workflow_database_path import (
    load_workflow_database_path,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_agent_audit_repository as audit_repository_module,
)
from agent_team.infrastructure.persistence.sqlite.audit import (
    sqlite_audit_migration_error,
)


def main(
    argv: Sequence[str] | None = None,
    service: AuditQueryService | None = None,
) -> int:
    """Run the human audit command-line interface."""
    arguments = _parse_arguments(argv)

    try:
        query_service = service or build_audit_query_service()
        if arguments.command == "list-runs":
            _print_runs(query_service.list_runs(arguments.limit))
        elif arguments.command == "show-run":
            _print_run_details(query_service.show_run(arguments.run_id))
        else:
            raise RuntimeError("Unsupported audit command.")
    except AgentRunNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    except sqlite_audit_migration_error.SQLiteAuditMigrationError as error:
        print(f"Database migration failed: {error}", file=sys.stderr)
        return 1

    return 0


def build_audit_query_service() -> AuditQueryService:
    """Build the audit query service with local SQLite persistence."""
    return AuditQueryService(
        reader=audit_repository_module.SQLiteAgentAuditRepository(
            load_workflow_database_path(),
        ),
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-team-audit")
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_runs = subcommands.add_parser("list-runs")
    list_runs.add_argument(
        "--limit",
        type=_positive_integer,
        default=10,
    )

    show_run = subcommands.add_parser("show-run")
    show_run.add_argument("run_id", type=_positive_integer)
    return parser.parse_args(argv)


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer",
        ) from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _print_runs(runs: Sequence[AgentRunRecord]) -> None:
    if not runs:
        print("No agent runs found.")
        return

    print("ID\tSTATUS\tROLE\tMODEL\tSTARTED\tPROMPT")
    for run in runs:
        print(
            "\t".join(
                (
                    str(run.id),
                    run.status.value,
                    run.role.value,
                    run.model,
                    _timestamp(run.started_at),
                    run.prompt_excerpt,
                ),
            ),
        )


def _print_run_details(details: AgentRunDetails) -> None:
    run = details.run
    print(f"Run {run.id}")
    print(f"Status: {run.status.value}")
    print(f"Role: {run.role.value}")
    print(f"Model: {run.model}")
    print(f"Feature ID: {_optional_number(run.feature_id)}")
    print(f"Session ID: {_optional_text(run.session_id)}")
    print(f"Started: {_timestamp(run.started_at)}")
    print(f"Ended: {_optional_timestamp(run.ended_at)}")
    print(f"Max turns: {run.max_turns}")
    print(f"Prompt: {run.prompt_excerpt}")
    print(f"Output: {_optional_text(run.output_excerpt)}")
    _print_generation_metadata(run)
    if run.error_type is not None or run.error_message is not None:
        print(
            "Error: "
            f"{_optional_text(run.error_type)} "
            f"{_optional_text(run.error_message)}",
        )
    _print_tool_invocations(details.tool_invocations)


def _print_generation_metadata(run: AgentRunRecord) -> None:
    metadata = run.generation_metadata
    if metadata is None:
        print("Generation metadata: -")
        return
    print("Generation metadata:")
    print(f"  Finish reason: {_optional_text(metadata.finish_reason)}")
    print(f"  Input tokens: {_optional_number(metadata.input_tokens)}")
    print(f"  Output tokens: {_optional_number(metadata.output_tokens)}")
    print(f"  Visible output chars: {metadata.visible_output_char_count}")
    print(f"  Objectively truncated: {metadata.objectively_truncated}")
    print(f"  Model: {metadata.model}")


def _print_tool_invocations(
    invocations: Sequence[ToolInvocationRecord],
) -> None:
    print("Tool invocations:")
    if not invocations:
        print("- none")
        return
    for invocation in invocations:
        print(
            "- "
            f"{invocation.id} {invocation.status.value} "
            f"{invocation.server_name}.{invocation.tool_name} "
            f"({invocation.classification.value})",
        )
        print(f"  Started: {_timestamp(invocation.started_at)}")
        print(f"  Ended: {_optional_timestamp(invocation.ended_at)}")
        print(f"  Arguments: {invocation.arguments_preview_json}")
        print(f"  Result: {_optional_text(invocation.result_preview)}")
        if (
            invocation.error_type is not None
            or invocation.error_message is not None
        ):
            print(
                "  Error: "
                f"{_optional_text(invocation.error_type)} "
                f"{_optional_text(invocation.error_message)}",
            )


def _timestamp(value: datetime) -> str:
    return value.isoformat()


def _optional_timestamp(value: datetime | None) -> str:
    if value is None:
        return "-"
    return _timestamp(value)


def _optional_text(value: str | None) -> str:
    if value is None:
        return "-"
    return value


def _optional_number(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
