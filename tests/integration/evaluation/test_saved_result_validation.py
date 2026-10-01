"""Preserve evaluation evidence types when reloading local saved results."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from agent_team.infrastructure.evaluation.json_eval_result_repository import (
    JsonEvalResultRepository,
)


class TestSavedResultValidation:
    """Reject corrupted persisted evidence instead of coercing its meaning."""

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        (
            (("case_results",), dict[str, object](), "object list"),
            (("case_results",), ["bad"], "JSON object"),
            (("warnings",), "bad", "string list"),
            (("warnings",), [False], "string"),
            (("duration_seconds",), "long", "number"),
            (("candidate_thinking_enabled",), "false", "boolean"),
            (("judge_thinking_enabled",), "false", "boolean"),
            (("case_results", 0, "repetition"), "one", "integer"),
            (
                ("case_results", 0, "deterministic_grade", "passed"),
                "true",
                "boolean",
            ),
            (
                ("case_results", 0, "candidate_result", "max_output_tokens"),
                "many",
                "integer",
            ),
            (
                ("case_results", 0, "candidate_result", "tool_calls"),
                [{"name": "read_file", "arguments": list[object]()}],
                "JSON object",
            ),
        ),
    )
    def test_rejects_corrupted_saved_results(
        self,
        tmp_path: Path,
        evaluation_result_record: dict[str, object],
        path: tuple[str | int, ...],
        value: object,
        message: str,
    ) -> None:
        """Keep malformed diagnostic fields from becoming valid evidence."""
        target: object = evaluation_result_record
        for key in path[:-1]:
            if isinstance(key, str):
                target = cast("Mapping[str, object]", target)[key]
            else:
                target = cast("Sequence[object]", target)[key]
        key = path[-1]
        assert isinstance(key, str)
        cast("dict[str, object]", target)[key] = value
        saved = tmp_path / "boundary-run.json"
        saved.write_text(
            json.dumps(evaluation_result_record), encoding="utf-8"
        )

        with pytest.raises(ValueError, match=message):
            JsonEvalResultRepository(tmp_path).get("boundary-run")

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("confidence", "certain", "number"),
            ("scores", {"correctness": "four"}, "integer"),
            ("reasons", {"correctness": False}, "string"),
            ("validation_errors", "bad", "string list"),
            ("retry_count", "one", "integer"),
        ),
    )
    def test_rejects_malformed_judge_grade(
        self,
        tmp_path: Path,
        evaluation_result_record: dict[str, object],
        field: str,
        value: object,
        message: str,
    ) -> None:
        """A judge score must remain numeric when reports are reloaded."""
        cases = cast(
            "list[dict[str, object]]", evaluation_result_record["case_results"]
        )
        cases[0]["judge_grade"] = {
            "verdict": "pass",
            "scores": {"correctness": 4},
            "reasons": {"correctness": "Correct."},
            "confidence": 1.0,
            "ambiguous": False,
        }
        cast("dict[str, object]", cases[0]["judge_grade"])[field] = value
        (tmp_path / "boundary-run.json").write_text(
            json.dumps(evaluation_result_record), encoding="utf-8"
        )

        with pytest.raises(ValueError, match=message):
            JsonEvalResultRepository(tmp_path).get("boundary-run")
