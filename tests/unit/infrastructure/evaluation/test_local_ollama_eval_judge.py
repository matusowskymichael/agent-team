"""Tests for the local Ollama eval judge adapter."""

import asyncio
import importlib
import json
from pathlib import Path
from typing import cast

import pytest

from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.judge_correction_request import (
    JudgeCorrectionRequest,
)
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.observed_tool_call import ObservedToolCall
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.infrastructure.evaluation.jsonl_golden_dataset_loader import (
    JsonlGoldenDatasetLoader,
)
from agent_team.infrastructure.evaluation.local_ollama_eval_judge import (
    LocalOllamaEvalJudge,
)
from agent_team.infrastructure.evaluation.markdown_rubric_loader import (
    MarkdownRubricLoader,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings

local_ollama_eval_judge = importlib.import_module(
    "agent_team.infrastructure.evaluation.local_ollama_eval_judge",
)


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return _Response(self.contents.pop(0))


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = _Chat(completions)


class TestLocalOllamaEvalJudge:
    """LocalOllamaEvalJudge behavior tests."""

    def test_parses_valid_local_judge_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return a validated judge grade from JSON output."""
        completions = _Completions(_valid_judge_json())

        def create_client(_settings: OllamaSettings) -> _Client:
            return _Client(completions)

        monkeypatch.setattr(
            local_ollama_eval_judge,
            "create_ollama_openai_client",
            create_client,
        )

        grade = asyncio.run(
            LocalOllamaEvalJudge(OllamaSettings()).grade(
                _first_case(),
                _rubric(),
                CandidateRunResult(
                    role=DevelopmentRole.BUSINESS_ANALYST,
                    model="qwen3.5:9b",
                    final_response="Login",
                    tool_calls=(),
                    database_effects=(),
                ),
                "judge:local",
            ),
        )

        assert grade.verdict is EvalVerdict.PASS
        assert grade.scores["least_privilege"] == 4
        assert grade.evidence["clarity"] == "observable"
        assert grade.rubric_id == _rubric().id
        assert completions.calls[0]["model"] == "judge:local"
        assert completions.calls[0]["temperature"] == 0
        assert "tools" not in completions.calls[0]

    def test_accepts_single_markdown_fenced_json_object(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Normalize one fenced JSON object safely."""
        completions = _Completions(f"```json\n{_valid_judge_json()}\n```")
        _patch_client(monkeypatch, completions)

        grade = asyncio.run(_grade())

        assert grade.verdict is EvalVerdict.PASS
        assert grade.validation_errors == ()

    def test_rejects_substantive_prose_outside_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject leading or trailing prose around JSON."""
        completions = _Completions(
            f"Here is the result: {_valid_judge_json()}",
        )
        _patch_client(monkeypatch, completions)

        grade = asyncio.run(_grade())

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert "prose outside JSON" in grade.validation_errors[0]

    def test_rejects_multiple_json_objects(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject multiple JSON objects in one final answer."""
        completions = _Completions(f"{_valid_judge_json()}\n{{}}")
        _patch_client(monkeypatch, completions)

        grade = asyncio.run(_grade())

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert "multiple JSON objects" in grade.validation_errors[0]

    def test_rejects_non_integer_score(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject score values that are not integers."""
        invalid_json = _valid_judge_json().replace(
            '"factual_grounding":4',
            '"factual_grounding":3.5',
        )
        completions = _Completions(invalid_json)
        _patch_client(monkeypatch, completions)

        grade = asyncio.run(_grade())

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert "scores.factual_grounding" in grade.validation_errors[0]

    def test_invalid_json_returns_judge_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid judge JSON is recorded as a judge error."""
        completions = _Completions("not-json")
        _patch_client(monkeypatch, completions)

        grade = asyncio.run(_grade())

        assert grade.verdict is EvalVerdict.JUDGE_ERROR
        assert grade.ambiguous is True
        assert grade.response_hash is not None
        assert grade.response_preview == "not-json"

    def test_correct_grade_sends_schema_correction_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Send one correction request through local Ollama."""
        completions = _Completions(_valid_judge_json())
        _patch_client(monkeypatch, completions)

        grade = asyncio.run(
            LocalOllamaEvalJudge(OllamaSettings()).correct_grade(
                _first_case(),
                _rubric(),
                _candidate(),
                "judge:local",
                JudgeCorrectionRequest(
                    invalid_response="not-json",
                    validation_errors=("root: invalid JSON",),
                ),
            ),
        )

        messages = completions.calls[0]["messages"]
        assert isinstance(messages, list)
        user_message = cast("dict[str, object]", messages[1])
        content = user_message.get("content")
        assert isinstance(content, str)
        assert "invalid_final_answer" in content
        assert grade.verdict is EvalVerdict.PASS

    def test_context_only_payload_identifies_matched_trajectory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tell the judge that ba-dev-014 matched context-only."""
        payload = _payload_for(
            case=_case("ba-dev-014"),
            candidate=_ba_dev_014_context_only_candidate(),
            monkeypatch=monkeypatch,
        )

        contract = _object_value(payload, "tool_contract")
        matched = _object_value(contract, "matched_acceptable_trajectory")
        assert payload["case_intent"] == "outcome_grounding"
        assert payload["evaluation_context_policy"] == (
            "standard_feature_context"
        )
        assert contract["acceptable_trajectory_matched"] is True
        assert contract["matched_trajectory_context_only"] is True
        assert contract["deterministic_passed"] is True
        assert contract["deterministic_hard_gate_failed"] is False
        assert contract["observed_tool_trajectory"] == []
        assert matched["required_tool_calls"] == []

    def test_context_only_response_is_grounded_in_authoritative_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Include feature-scoped context that grounded the response."""
        payload = _payload_for(
            case=_case("ba-dev-014"),
            candidate=_ba_dev_014_context_only_candidate(),
            monkeypatch=monkeypatch,
        )

        context = _object_value(payload, "authoritative_context")
        context_json = json.dumps(context, sort_keys=True)
        assert payload["candidate_response"] == _ba_dev_014_response()
        assert context["available_to_candidate"] is True
        assert context["feature_scope_id"] == 1
        assert "FeatureContextBuilder" in str(context["source"])
        assert "First-only requirement" in context_json
        assert "Second-only secret" not in context_json
        assert "authoritative evidence" in str(context["grounding_rule"])

    def test_no_preload_context_hides_metadata_from_judge_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tell the judge when metadata retrieval required a tool."""
        payload = _payload_for(
            case=_case("ba-dev-002"),
            candidate=CandidateRunResult(
                role=DevelopmentRole.BUSINESS_ANALYST,
                model="qwen3.5:9b",
                final_response="Billing. Invoices and receipts.",
                tool_calls=(
                    ObservedToolCall(
                        name="get_feature",
                        arguments={"feature_id": 1},
                        status="completed",
                    ),
                ),
                database_effects=(),
            ),
            monkeypatch=monkeypatch,
        )

        context = _object_value(payload, "authoritative_context")
        context_json = json.dumps(context, sort_keys=True)
        contract = _object_value(payload, "tool_contract")
        assert payload["case_intent"] == "tool_dispatch"
        assert payload["evaluation_context_policy"] == "no_feature_preload"
        assert context["available_to_candidate"] is False
        assert context["feature"] is None
        assert "Billing" not in context_json
        assert "Invoices and receipts" not in context_json
        assert contract["deterministic_passed"] is True

    def test_metadata_only_context_hides_artifact_content_from_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keep artifact-dispatch facts out of metadata-only context."""
        payload = _payload_for(
            case=_case("ba-dev-004"),
            candidate=CandidateRunResult(
                role=DevelopmentRole.BUSINESS_ANALYST,
                model="qwen3.5:9b",
                final_response="Search by keyword.",
                tool_calls=(
                    ObservedToolCall(
                        name="list_artifacts",
                        arguments={"feature_id": 1},
                        status="completed",
                    ),
                ),
                database_effects=(),
            ),
            monkeypatch=monkeypatch,
        )

        context = _object_value(payload, "authoritative_context")
        feature = _object_value(context, "feature")
        context_json = json.dumps(context, sort_keys=True)
        assert payload["case_intent"] == "tool_dispatch"
        assert payload["evaluation_context_policy"] == (
            "metadata_only_feature_context"
        )
        assert context["available_to_candidate"] is True
        assert feature["title"] == "Search"
        assert context["artifacts"] == []
        assert "Search by keyword" not in context_json
        assert "Use an index" not in context_json

    def test_judge_payload_does_not_make_legacy_path_mandatory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keep legacy tool expectations diagnostic after a match."""
        completions = _Completions(_valid_judge_json())
        _patch_client(monkeypatch, completions)

        asyncio.run(
            LocalOllamaEvalJudge(OllamaSettings()).grade(
                _case("ba-dev-014"),
                _rubric(),
                _ba_dev_014_context_only_candidate(),
                "judge:local",
            ),
        )

        payload = _payload_from_completion_call(completions)
        system_prompt = _system_prompt_from_completion_call(completions)
        expected_calls = _object_list_value(payload, "expected_tool_calls")
        assert expected_calls[0]["name"] == "list_artifacts"
        assert "not mandatory" in str(payload["expected_tool_calls_note"])
        assert "Do not treat legacy/default expected tool calls" in (
            system_prompt
        )
        assert "when another acceptable trajectory matched" in system_prompt

    def test_context_only_fails_when_no_empty_trajectory_exists(self) -> None:
        """Require a tool when the deterministic contract requires one."""
        grade = DeterministicEvalGrader().grade(
            _case("ba-dev-015"),
            CandidateRunResult(
                role=DevelopmentRole.BUSINESS_ANALYST,
                model="qwen3.5:9b",
                final_response="Invoices include tax.",
                tool_calls=(),
                database_effects=(),
            ),
            "qwen3.5:9b",
        )

        assert grade.passed is False
        assert "missing acceptable tool trajectory" in grade.reasons

    def test_feature_two_information_remains_prohibited(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keep cross-feature leakage visible as a hard boundary."""
        payload = _payload_for(
            case=_case("ba-dev-014"),
            candidate=_ba_dev_014_context_only_candidate(),
            monkeypatch=monkeypatch,
        )

        context_json = json.dumps(
            payload["authoritative_context"],
            sort_keys=True,
        )
        prohibited_claims = _string_list_value(
            payload,
            "prohibited_objective_claims",
        )
        assert "Second-only secret" in prohibited_claims
        assert "Second-only secret" not in context_json

    def test_judge_instructions_still_penalize_unsupported_facts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Do not turn accepted context-only into blanket acceptance."""
        completions = _Completions(_valid_judge_json())
        _patch_client(monkeypatch, completions)

        asyncio.run(
            LocalOllamaEvalJudge(OllamaSettings()).grade(
                _case("ba-dev-014"),
                _rubric(),
                _ba_dev_014_context_only_candidate(),
                "judge:local",
            ),
        )

        system_prompt = _system_prompt_from_completion_call(completions)
        assert "Unsupported claims absent from authoritative context" in (
            system_prompt
        )
        assert "Deterministic hard gates are authoritative" in system_prompt

    def test_tool_required_case_payload_remains_mandatory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Do not treat context-only as accepted for ordinary tool cases."""
        payload = _payload_for(
            case=_case("ba-dev-003"),
            candidate=CandidateRunResult(
                role=DevelopmentRole.BUSINESS_ANALYST,
                model="qwen3.5:9b",
                final_response=(
                    "Users can edit display names. The task collection is "
                    "empty."
                ),
                tool_calls=(),
                database_effects=(),
            ),
            monkeypatch=monkeypatch,
        )

        contract = _object_value(payload, "tool_contract")
        expected_calls = _object_list_value(payload, "expected_tool_calls")
        assert contract["acceptable_trajectory_matched"] is False
        assert contract["matched_trajectory_context_only"] is False
        assert expected_calls[0]["name"] == "get_feature_overview"
        reasons = _string_list_value(
            contract,
            "deterministic_failure_reasons",
        )
        assert "missing expected tool call get_feature_overview" in (reasons)

    def test_observed_read_only_trajectory_is_reported(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Show the judge actual observed tool calls, not just defaults."""
        payload = _payload_for(
            case=_case("ba-dev-014"),
            candidate=CandidateRunResult(
                role=DevelopmentRole.BUSINESS_ANALYST,
                model="qwen3.5:9b",
                final_response=_ba_dev_014_response(),
                tool_calls=(
                    ObservedToolCall(
                        name="get_feature_overview",
                        arguments={"feature_id": 1},
                        status="completed",
                    ),
                ),
                database_effects=(),
            ),
            monkeypatch=monkeypatch,
        )

        contract = _object_value(payload, "tool_contract")
        observed = _object_list_value(contract, "observed_tool_trajectory")
        matched = _object_value(contract, "matched_acceptable_trajectory")
        required = _object_list_value(matched, "required_tool_calls")
        assert contract["acceptable_trajectory_matched"] is True
        assert contract["matched_trajectory_context_only"] is False
        assert observed[0]["name"] == "get_feature_overview"
        assert required[0]["name"] == "get_feature_overview"


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    completions: _Completions,
) -> None:
    def create_client(_settings: OllamaSettings) -> _Client:
        return _Client(completions)

    monkeypatch.setattr(
        local_ollama_eval_judge,
        "create_ollama_openai_client",
        create_client,
    )


async def _grade() -> JudgeGrade:
    return await LocalOllamaEvalJudge(OllamaSettings()).grade(
        _first_case(),
        _rubric(),
        _candidate(),
        "judge:local",
    )


def _payload_for(
    case: EvalCase,
    candidate: CandidateRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    completions = _Completions(_valid_judge_json())
    _patch_client(monkeypatch, completions)
    asyncio.run(
        LocalOllamaEvalJudge(OllamaSettings()).grade(
            case,
            _rubric(),
            candidate,
            "judge:local",
        ),
    )
    return _payload_from_completion_call(completions)


def _payload_from_completion_call(
    completions: _Completions,
) -> dict[str, object]:
    messages = _messages_from_completion_call(completions)
    content = messages[1]["content"]
    assert isinstance(content, str)
    return cast("dict[str, object]", json.loads(content))


def _system_prompt_from_completion_call(completions: _Completions) -> str:
    messages = _messages_from_completion_call(completions)
    content = messages[0]["content"]
    assert isinstance(content, str)
    return content


def _messages_from_completion_call(
    completions: _Completions,
) -> list[dict[str, object]]:
    messages = completions.calls[0]["messages"]
    assert isinstance(messages, list)
    return cast("list[dict[str, object]]", messages)


def _object_value(
    mapping: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = mapping[key]
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _object_list_value(
    mapping: dict[str, object],
    key: str,
) -> list[dict[str, object]]:
    value = mapping[key]
    assert isinstance(value, list)
    return cast("list[dict[str, object]]", value)


def _string_list_value(
    mapping: dict[str, object],
    key: str,
) -> list[str]:
    value = mapping[key]
    assert isinstance(value, list)
    return cast("list[str]", value)


def _ba_dev_014_context_only_candidate() -> CandidateRunResult:
    return CandidateRunResult(
        role=DevelopmentRole.BUSINESS_ANALYST,
        model="qwen3.5:9b",
        final_response=_ba_dev_014_response(),
        tool_calls=(),
        database_effects=(),
    )


def _ba_dev_014_response() -> str:
    return (
        'Based on the current workflow data for **Feature 1** ("Feature '
        'One"), there is one attached requirements artifact:\n\n'
        "- **Artifact ID 1**: `First-only requirement.`\n\n"
        "This represents the only documented requirement for this feature "
        "at present."
    )


def _candidate() -> CandidateRunResult:
    return CandidateRunResult(
        role=DevelopmentRole.BUSINESS_ANALYST,
        model="qwen3.5:9b",
        final_response="Login",
        tool_calls=(),
        database_effects=(),
    )


def _valid_judge_json() -> str:
    rubric = _rubric()
    dimensions = {dimension.id: 4 for dimension in rubric.dimensions}
    text_by_dimension = {
        dimension.id: "observable" for dimension in rubric.dimensions
    }
    return (
        "{"
        f'"rubric_id":"{rubric.id}",'
        f'"rubric_version":"{rubric.version}",'
        '"case_id":"ba-dev-001",'
        f'"scores":{_json(dimensions)},'
        f'"reasons":{_json(text_by_dimension)},'
        f'"evidence":{_json(text_by_dimension)},'
        '"verdict":"pass",'
        '"confidence":0.9,'
        '"ambiguous":false'
        "}"
    )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _first_case() -> EvalCase:
    suite = JsonlGoldenDatasetLoader().load(
        "business_analyst_development",
        Path("evals/datasets/business_analyst_development.jsonl"),
    )
    return suite.cases[0]


def _case(case_id: str) -> EvalCase:
    suite = JsonlGoldenDatasetLoader().load(
        "business_analyst_development",
        Path("evals/datasets/business_analyst_development.jsonl"),
    )
    return next(case for case in suite.cases if case.id == case_id)


def _rubric() -> Rubric:
    return MarkdownRubricLoader().load(
        Path("evals/rubrics/business_analyst.md"),
    )
