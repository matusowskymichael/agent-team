"""Fail closed on malformed records without changing golden datasets."""

import json
from pathlib import Path

import pytest

from agent_team.infrastructure.evaluation.jsonl_golden_dataset_loader import (
    JsonlGoldenDatasetLoader,
)


class TestDatasetValidation:
    """Keep expected tool calls and trusted fixture shapes deterministic."""

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("id", 17, "string value"),
            ("feature_fixtures", dict[str, object](), "list of objects"),
            ("feature_fixtures", ["bad"], "JSON object"),
            ("feature_fixtures", [{"id": "one"}], "integer value"),
            ("prior_session_turns", "bad", "list of strings"),
            ("prior_session_turns", [False], "string value"),
            ("task_scope_id", "one", "integer value"),
            ("semantic_judge_required", "false", "boolean value"),
            ("expected_error", "bad", "JSON object"),
            (
                "expected_database_effects",
                [
                    {
                        "table": "tasks",
                        "operation": "update",
                        "field_values": list[object](),
                    }
                ],
                "object value",
            ),
        ),
    )
    def test_rejects_malformed_dataset_records(
        self,
        tmp_path: Path,
        evaluation_case_record: dict[str, object],
        field: str,
        value: object,
        message: str,
    ) -> None:
        """Reject corrupt fixture or expectation types before execution."""
        evaluation_case_record[field] = value
        path = tmp_path / "case.jsonl"
        path.write_text(json.dumps(evaluation_case_record), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            JsonlGoldenDatasetLoader().load("boundary", path)

    @pytest.mark.parametrize("text", ("[]", "broken-json"))
    def test_rejects_non_object_records_with_source_location(
        self, tmp_path: Path, text: str
    ) -> None:
        """Report the input record location when JSON cannot be loaded."""
        path = tmp_path / "case.jsonl"
        path.write_text("\n" + text, encoding="utf-8")

        with pytest.raises(ValueError, match=r"case.jsonl:2"):
            JsonlGoldenDatasetLoader().load("boundary", path)

    def test_loads_unversioned_metadata_and_flat_expected_errors(
        self,
        tmp_path: Path,
        evaluation_case_record: dict[str, object],
    ) -> None:
        """Preserve older local evaluation record formats and exact symbols."""
        evaluation_case_record.update(
            expected_error_type="AgentStalledError",
            expected_error_stage="candidate_execution",
            expected_error_message_code="no new successful",
        )
        evaluation_case_record.pop("expected_error", None)
        path = tmp_path / "case.jsonl"
        path.write_text(
            '{"record_type":"dataset_metadata"}\n\n'
            + json.dumps(evaluation_case_record),
            encoding="utf-8",
        )

        suite = JsonlGoldenDatasetLoader().load("boundary", path)

        assert suite.dataset_version is None
        case = suite.cases[0]
        assert case.expected_error is not None
        assert case.expected_error.error_type == "AgentStalledError"
        assert case.expected_error.message_fragment == "no new successful"
        assert any(
            call.name == "find_symbol"
            and call.arguments_subset.get("name") == "AuthService.logout"
            for call in case.expected_tool_calls
        )
