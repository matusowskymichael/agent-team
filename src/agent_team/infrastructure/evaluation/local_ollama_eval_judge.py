"""Local Ollama evaluation judge adapter."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from typing import cast

from agents import set_tracing_disabled

from agent_team.application.audit.audit_sanitizer import (
    hash_text,
    sanitize_text,
)
from agent_team.application.evaluation.deterministic_eval_grader import (
    DeterministicEvalGrader,
)
from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.eval_artifact_fixture import (
    EvalArtifactFixture,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_context_policy import (
    EvalContextPolicy,
)
from agent_team.domain.evaluation.eval_feature_fixture import (
    EvalFeatureFixture,
)
from agent_team.domain.evaluation.eval_task_fixture import EvalTaskFixture
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)
from agent_team.domain.evaluation.judge_correction_request import (
    JudgeCorrectionRequest,
)
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.observed_tool_call import ObservedToolCall
from agent_team.domain.evaluation.rubric import Rubric
from agent_team.domain.evaluation.rubric_dimension import RubricDimension
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.infrastructure.ollama.ollama_model_factory import (
    create_ollama_openai_client,
)
from agent_team.infrastructure.ollama.ollama_model_settings import (
    create_ollama_chat_extra_body,
)
from agent_team.infrastructure.ollama.ollama_settings import OllamaSettings

MAX_JUDGE_COMPLETION_TOKENS = 1800
_JudgeResponseMetadata = tuple[str, str, str]

_FENCED_JSON_PATTERN = re.compile(
    r"\A```(?:json)?\s*(?P<body>.*?)\s*```\Z",
    re.IGNORECASE | re.DOTALL,
)
_THINKING_PATTERN = re.compile(
    r"<think>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)

_ROLES_WITH_ALL_TASK_CONTEXT = frozenset(
    {
        DevelopmentRole.DELIVERY_MANAGER,
        DevelopmentRole.SOFTWARE_ARCHITECT,
        DevelopmentRole.QA_ENGINEER,
        DevelopmentRole.CODE_REVIEWER,
    },
)

_VISIBLE_ARTIFACT_KINDS_BY_ROLE = {
    DevelopmentRole.BUSINESS_ANALYST: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
        },
    ),
    DevelopmentRole.SOFTWARE_ARCHITECT: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        },
    ),
    DevelopmentRole.BACKEND_DEVELOPER: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        },
    ),
    DevelopmentRole.FRONTEND_DEVELOPER: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
        },
    ),
    DevelopmentRole.QA_ENGINEER: frozenset(
        {
            ArtifactKind.REQUIREMENTS,
            ArtifactKind.ACCEPTANCE_CRITERIA,
            ArtifactKind.ARCHITECTURE,
            ArtifactKind.IMPLEMENTATION_PLAN,
            ArtifactKind.TEST_REPORT,
        },
    ),
    DevelopmentRole.CODE_REVIEWER: frozenset(ArtifactKind),
    DevelopmentRole.DELIVERY_MANAGER: frozenset(ArtifactKind),
}


@dataclass(frozen=True, slots=True)
class LocalOllamaEvalJudge:
    """Local Ollama-backed rubric judge with no workflow tools."""

    settings: OllamaSettings

    async def grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
    ) -> JudgeGrade:
        """Judge one candidate result using local OpenAI-compatible chat."""
        return await self._complete_and_parse(
            rubric=rubric,
            judge_model=judge_model,
            user_content=_user_prompt(case, candidate, rubric),
        )

    async def correct_grade(
        self,
        case: EvalCase,
        rubric: Rubric,
        candidate: CandidateRunResult,
        judge_model: str,
        correction: JudgeCorrectionRequest,
    ) -> JudgeGrade:
        """Correct invalid judge output using local Ollama."""
        return await self._complete_and_parse(
            rubric=rubric,
            judge_model=judge_model,
            user_content=_correction_prompt(
                case=case,
                rubric=rubric,
                candidate=candidate,
                correction=correction,
            ),
        )

    async def _complete_and_parse(
        self,
        rubric: Rubric,
        judge_model: str,
        user_content: str,
    ) -> JudgeGrade:
        """Request one local completion and parse the final answer."""
        set_tracing_disabled(True)
        judge_settings = OllamaSettings(
            base_url=self.settings.base_url,
            model=judge_model,
            max_output_tokens=self.settings.max_output_tokens,
            thinking_enabled=self.settings.thinking_enabled,
        )
        client = create_ollama_openai_client(judge_settings)
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "system",
                    "content": _system_prompt(rubric),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0,
            max_completion_tokens=MAX_JUDGE_COMPLETION_TOKENS,
            extra_body=create_ollama_chat_extra_body(judge_settings),
        )
        content = response.choices[0].message.content or ""
        return _parse_grade(content, judge_model)


def _system_prompt(rubric: Rubric) -> str:
    return "\n".join(
        (
            "You are a local evaluation judge.",
            "Return exactly one JSON object and no other text.",
            "Do not use Markdown fences unless correcting a fenced answer.",
            "Do not include hidden reasoning or chain-of-thought.",
            "Judge only observable candidate output and recorded tool data.",
            "Deterministic hard gates are authoritative.",
            "Preloaded authoritative feature context counts as grounded "
            "evidence.",
            "When deterministic evaluation confirms a matched context-only "
            "trajectory, do not demand a tool call.",
            "A context-only response must not lose factual-grounding, "
            "tool-accuracy, least-privilege, or uncertainty points merely "
            "because no tool was called when context-only is accepted.",
            "Do not treat legacy/default expected tool calls as mandatory "
            "when another acceptable trajectory matched.",
            "Unnecessary tool calls may reduce least-privilege or "
            "tool-accuracy scores.",
            "Unsupported claims absent from authoritative context and tool "
            "results must still be penalized.",
            "Context-only is unacceptable when the deterministic contract "
            "requires a tool call.",
            "For outcome_grounding cases, a response may correctly use "
            "authoritative context when context-only is explicitly accepted.",
            "For tool_dispatch cases, the candidate must follow the declared "
            "retrieval contract even if a similar answer could be inferred.",
            "Context-only is valid only when the golden case explicitly "
            "declares an empty accepted trajectory.",
            "Deterministic trajectory validation remains authoritative.",
            "Use the schema and rubric exactly.",
            _schema_text(rubric),
        ),
    )


def _schema_text(rubric: Rubric) -> str:
    return json.dumps(_schema_payload(rubric), sort_keys=True)


def _schema_payload(rubric: Rubric) -> dict[str, object]:
    dimensions = [dimension.id for dimension in rubric.dimensions]
    return {
        "rubric_id": rubric.id,
        "rubric_version": rubric.version,
        "case_id": "selected case ID",
        "scores": dict.fromkeys(dimensions, "integer 0..4"),
        "reasons": dict.fromkeys(dimensions, "concise non-empty reason"),
        "evidence": dict.fromkeys(dimensions, "observable evidence only"),
        "verdict": "pass or fail",
        "confidence": "number 0..1",
        "ambiguous": "boolean",
    }


def _user_prompt(
    case: EvalCase,
    candidate: CandidateRunResult,
    rubric: Rubric,
) -> str:
    tool_contract = _tool_contract_payload(case, candidate)
    payload = {
        "case_id": case.id,
        "rubric_id": rubric.id,
        "rubric_version": rubric.version,
        "case_intent": case.intent.value,
        "evaluation_context_policy": case.context_policy.value,
        "rubric_dimensions": [
            _dimension_payload(dimension) for dimension in rubric.dimensions
        ],
        "user_input": case.user_input,
        "tool_contract": tool_contract,
        "expected_tool_calls": tool_contract[
            "legacy_default_expected_tool_calls"
        ],
        "expected_tool_calls_note": tool_contract[
            "legacy_default_expected_tool_calls_note"
        ],
        "acceptable_tool_trajectories": tool_contract[
            "acceptable_tool_trajectories"
        ],
        "matched_acceptable_trajectory": tool_contract[
            "matched_acceptable_trajectory"
        ],
        "matched_trajectory_context_only": tool_contract[
            "matched_trajectory_context_only"
        ],
        "forbidden_tool_calls": list(case.forbidden_tool_calls),
        "expected_database_effects": [
            {
                "table": effect.table,
                "operation": effect.operation,
                "field_values": effect.field_values,
            }
            for effect in case.expected_database_effects
        ],
        "objective_response_facts": list(case.objective_response_facts),
        "semantic_response_requirements": list(
            case.semantic_response_requirements,
        ),
        "prohibited_objective_claims": list(
            case.prohibited_objective_claims,
        ),
        "semantic_judge_required": case.semantic_judge_required,
        "authoritative_context": _authoritative_context_payload(case),
        "candidate_response": candidate.final_response,
        "observed_tool_calls": _observed_tool_calls_payload(candidate),
        "observed_tool_trajectory": [
            call.name for call in candidate.tool_calls
        ],
        "observed_skill_calls": _observed_skill_calls_payload(candidate),
        "observed_database_effects": [
            {
                "table": effect.table,
                "operation": effect.operation,
                "field_values": effect.field_values,
            }
            for effect in candidate.database_effects
        ],
    }
    return json.dumps(payload, sort_keys=True)


def _dimension_payload(dimension: RubricDimension) -> dict[str, object]:
    return {
        "id": dimension.id,
        "name": dimension.name,
        "weight": dimension.weight,
        "minimum_score": dimension.minimum_score,
        "critical": dimension.critical,
    }


def _correction_prompt(
    case: EvalCase,
    rubric: Rubric,
    candidate: CandidateRunResult,
    correction: JudgeCorrectionRequest,
) -> str:
    payload = {
        "task": "Correct the invalid judge JSON.",
        "case_id": case.id,
        "candidate_status": candidate.status,
        "required_schema": _schema_payload(rubric),
        "validation_errors": list(correction.validation_errors),
        "invalid_final_answer": correction.invalid_response,
        "instruction": "Return only the corrected JSON object.",
    }
    return json.dumps(payload, sort_keys=True)


def _tool_contract_payload(
    case: EvalCase,
    candidate: CandidateRunResult,
) -> dict[str, object]:
    deterministic_grade = DeterministicEvalGrader().grade(
        case=case,
        candidate=candidate,
        candidate_model=candidate.model,
    )
    matched_index = _matched_trajectory_index(
        case.acceptable_tool_trajectories,
        candidate.tool_calls,
    )
    matched_trajectory = (
        case.acceptable_tool_trajectories[matched_index]
        if matched_index is not None
        else None
    )
    matched_context_only = (
        matched_trajectory is not None
        and not matched_trajectory.required_tool_calls
        and not candidate.tool_calls
    )
    return {
        "deterministic_passed": deterministic_grade.passed,
        "deterministic_hard_gate_failed": (
            deterministic_grade.hard_gate_failed
        ),
        "deterministic_failure_reasons": list(deterministic_grade.reasons),
        "observed_tool_trajectory": _observed_tool_calls_payload(candidate),
        "legacy_default_expected_tool_calls": [
            _expected_tool_call_payload(call)
            for call in case.expected_tool_calls
        ],
        "legacy_default_expected_tool_calls_note": (
            "Legacy/default expected tool calls are retained for "
            "diagnostic compatibility. They are not mandatory when "
            "acceptable_trajectory_matched is true; judge the matched "
            "trajectory and deterministic hard-gate status instead. When "
            "acceptable_trajectory_matched is false, these calls describe "
            "the deterministic default contract."
        ),
        "acceptable_tool_trajectories": [
            _expected_tool_trajectory_payload(trajectory)
            for trajectory in case.acceptable_tool_trajectories
        ],
        "acceptable_trajectory_matched": matched_index is not None,
        "matched_acceptable_trajectory_index": matched_index,
        "matched_acceptable_trajectory": (
            _expected_tool_trajectory_payload(matched_trajectory)
            if matched_trajectory is not None
            else None
        ),
        "matched_trajectory_context_only": matched_context_only,
    }


def _authoritative_context_payload(case: EvalCase) -> dict[str, object]:
    feature_id = _feature_scope(case)
    feature = _feature_fixture(case, feature_id)
    if (
        feature is None
        or case.context_policy is EvalContextPolicy.NO_FEATURE_PRELOAD
    ):
        return {
            "available_to_candidate": False,
            "context_policy": case.context_policy.value,
            "source": (
                "Evaluation context policy did not preload feature workflow "
                "data for this case."
            ),
            "feature_scope_id": feature_id,
            "feature": None,
            "artifacts": [],
            "tasks": [],
            "omitted": ["feature", "artifacts", "tasks"],
            "grounding_rule": (
                "Only tool results and the candidate final response are "
                "available as grounding evidence for this case."
            ),
        }

    artifacts: tuple[EvalArtifactFixture, ...] = _visible_artifacts(
        case,
        feature,
    )
    tasks: tuple[EvalTaskFixture, ...] = _visible_tasks(case, feature)
    if case.context_policy is EvalContextPolicy.METADATA_ONLY_FEATURE_CONTEXT:
        artifacts = ()
        tasks = ()
        omitted = ["artifacts", "tasks"]
        source = (
            "Evaluation context policy preloaded feature metadata only before "
            "candidate execution."
        )
    else:
        omitted = _standard_omissions(case)
        source = (
            "FeatureContextBuilder preloaded authoritative feature-scoped "
            "workflow context before candidate execution."
        )

    return {
        "available_to_candidate": True,
        "context_policy": case.context_policy.value,
        "source": source,
        "feature_scope_id": feature_id,
        "feature": {
            "id": feature.id,
            "title": feature.title,
            "description": feature.description,
            "status": feature.status.value,
        },
        "artifacts": [
            {
                "feature_id": artifact.feature_id,
                "kind": artifact.kind.value,
                "content": artifact.content,
                "created_by": artifact.created_by,
            }
            for artifact in artifacts
        ],
        "tasks": [
            {
                "feature_id": task.feature_id,
                "title": task.title,
                "description": task.description,
                "assigned_role": task.assigned_role.value,
                "status": task.status.value,
            }
            for task in tasks
        ],
        "omitted": omitted,
        "grounding_rule": (
            "Facts present in this feature-scoped context count as "
            "authoritative evidence even when an accepted context-only "
            "trajectory matched."
        ),
    }


def _visible_artifacts(
    case: EvalCase,
    feature: EvalFeatureFixture,
) -> tuple[EvalArtifactFixture, ...]:
    visible_kinds = _visible_artifact_kinds(case.active_role)
    return tuple(
        artifact
        for artifact in feature.artifacts
        if artifact.kind in visible_kinds
    )


def _visible_tasks(
    case: EvalCase,
    feature: EvalFeatureFixture,
) -> tuple[EvalTaskFixture, ...]:
    if case.active_role in _ROLES_WITH_ALL_TASK_CONTEXT:
        return feature.tasks
    return ()


def _standard_omissions(case: EvalCase) -> list[str]:
    omitted: list[str] = []
    if case.active_role not in _ROLES_WITH_ALL_TASK_CONTEXT:
        omitted.append("tasks")
    hidden_kinds = [
        kind.value
        for kind in ArtifactKind
        if kind not in _visible_artifact_kinds(case.active_role)
    ]
    if hidden_kinds:
        omitted.append(f"artifact kinds: {', '.join(hidden_kinds)}")
    return omitted


def _visible_artifact_kinds(
    role: DevelopmentRole,
) -> frozenset[ArtifactKind]:
    return _VISIBLE_ARTIFACT_KINDS_BY_ROLE.get(role, frozenset(ArtifactKind))


def _feature_scope(case: EvalCase) -> int | None:
    if case.feature_scope_id is not None:
        return case.feature_scope_id
    if not case.feature_fixtures:
        return None
    return case.feature_fixtures[0].id


def _feature_fixture(
    case: EvalCase,
    feature_id: int | None,
) -> EvalFeatureFixture | None:
    if feature_id is None:
        return None
    return next(
        (
            feature
            for feature in case.feature_fixtures
            if feature.id == feature_id
        ),
        None,
    )


def _expected_tool_trajectory_payload(
    trajectory: ExpectedToolTrajectory,
) -> dict[str, object]:
    return {
        "required_tool_calls": [
            _expected_tool_call_payload(call)
            for call in trajectory.required_tool_calls
        ],
        "order_matters": trajectory.order_matters,
        "optional_read_only_tool_calls": list(
            trajectory.optional_read_only_tool_calls,
        ),
        "forbidden_tool_calls": list(trajectory.forbidden_tool_calls),
    }


def _expected_tool_call_payload(
    call: ExpectedToolCall,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": call.name,
        "arguments_subset": call.arguments_subset,
    }
    if call.order is not None:
        payload["order"] = call.order
    return payload


def _observed_tool_calls_payload(
    candidate: CandidateRunResult,
) -> list[dict[str, object]]:
    return [
        {
            "name": call.name,
            "arguments": call.arguments,
            "status": call.status,
            "reached_mcp": call.reached_mcp,
        }
        for call in candidate.tool_calls
    ]


def _observed_skill_calls_payload(
    candidate: CandidateRunResult,
) -> list[dict[str, object]]:
    return [
        {
            "tool_name": call.tool_name,
            "skill_name": call.skill_name,
            "status": call.status,
            "content_hash": call.content_hash,
            "resource_name": call.resource_name,
        }
        for call in candidate.skill_calls
    ]


def _matched_trajectory_index(
    trajectories: tuple[ExpectedToolTrajectory, ...],
    observed_calls: tuple[ObservedToolCall, ...],
) -> int | None:
    for index, trajectory in enumerate(trajectories):
        if _matches_trajectory(trajectory, observed_calls):
            return index
    return None


def _matches_trajectory(
    trajectory: ExpectedToolTrajectory,
    observed_calls: tuple[ObservedToolCall, ...],
) -> bool:
    if trajectory.order_matters:
        required_matches = _ordered_matches(
            trajectory.required_tool_calls,
            observed_calls,
        )
    else:
        required_matches = all(
            _has_expected_call(expected_call, observed_calls)
            for expected_call in trajectory.required_tool_calls
        )
    if not required_matches:
        return False

    allowed_names = {
        call.name for call in trajectory.required_tool_calls
    } | set(trajectory.optional_read_only_tool_calls)
    return all(call.name in allowed_names for call in observed_calls)


def _ordered_matches(
    expected_calls: tuple[ExpectedToolCall, ...],
    observed_calls: tuple[ObservedToolCall, ...],
) -> bool:
    observed_index = 0
    for expected_call in expected_calls:
        while observed_index < len(observed_calls):
            observed_call = observed_calls[observed_index]
            observed_index += 1
            if observed_call.name == expected_call.name and _contains_subset(
                observed_call.arguments,
                expected_call.arguments_subset,
            ):
                break
        else:
            return False
    return True


def _has_expected_call(
    expected_call: ExpectedToolCall,
    observed_calls: tuple[ObservedToolCall, ...],
) -> bool:
    return any(
        call.name == expected_call.name
        and _contains_subset(call.arguments, expected_call.arguments_subset)
        for call in observed_calls
    )


def _contains_subset(
    actual: dict[str, object],
    expected: dict[str, object],
) -> bool:
    return all(
        _contains_value(actual.get(key), value)
        for key, value in expected.items()
    )


def _contains_value(actual: object, expected: object) -> bool:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        actual_values = cast("Mapping[object, object]", actual)
        expected_values = cast("Mapping[object, object]", expected)
        return all(
            _contains_value(actual_values.get(key), value)
            for key, value in expected_values.items()
        )
    return actual == expected


def _parse_grade(
    content: str,
    judge_model: str,
) -> JudgeGrade:
    response_hash = hash_text(content)
    response_preview = _judge_preview(content)
    parsed, errors = _parse_json_object(content)
    if errors:
        return _judge_error(
            errors=errors,
            judge_model=judge_model,
            response_hash=response_hash,
            response_preview=response_preview,
            raw_response=content,
        )

    return _grade_from_mapping(
        data=parsed,
        judge_model=judge_model,
        response_metadata=(response_hash, response_preview, content),
    )


def _parse_json_object(
    content: str,
) -> tuple[Mapping[str, object], tuple[str, ...]]:
    text, errors = _normalized_json_text(content)
    if errors:
        return {}, errors
    return _decode_json_object(text)


def _normalized_json_text(content: str) -> tuple[str, tuple[str, ...]]:
    text = content.strip()
    fence = _FENCED_JSON_PATTERN.fullmatch(text)
    if fence is not None:
        return fence.group("body").strip(), ()
    if "```" in text:
        return "", ("root: invalid Markdown JSON fence",)
    if not text.startswith("{"):
        return "", ("root: substantive prose outside JSON object",)
    return text, ()


def _decode_json_object(
    text: str,
) -> tuple[Mapping[str, object], tuple[str, ...]]:
    decoder = json.JSONDecoder()
    try:
        parsed: object
        parsed, end_index = decoder.raw_decode(text)
    except JSONDecodeError as error:
        return {}, (f"root: invalid JSON at {error.pos}: {error.msg}",)

    remainder = text[end_index:].strip()
    if remainder.startswith("{"):
        return {}, ("root: multiple JSON objects",)
    if remainder:
        return {}, ("root: substantive prose outside JSON object",)
    if not isinstance(parsed, dict):
        return {}, ("root: expected JSON object",)
    return cast("Mapping[str, object]", parsed), ()


def _grade_from_mapping(
    data: Mapping[str, object],
    judge_model: str,
    response_metadata: _JudgeResponseMetadata,
) -> JudgeGrade:
    response_hash, response_preview, raw_response = response_metadata
    validation_errors: list[str] = []
    verdict_text = _text(data, "verdict", validation_errors)
    verdict = _verdict(verdict_text, validation_errors)
    grade = JudgeGrade(
        verdict=verdict,
        scores=_int_mapping(data, "scores", validation_errors),
        reasons=_str_mapping(data, "reasons", validation_errors),
        confidence=_float(data, "confidence", validation_errors),
        ambiguous=_bool(data, "ambiguous", validation_errors),
        rubric_id=_text(data, "rubric_id", validation_errors),
        rubric_version=_text(data, "rubric_version", validation_errors),
        case_id=_text(data, "case_id", validation_errors),
        evidence=_str_mapping(data, "evidence", validation_errors),
        judge_model=judge_model,
        response_hash=response_hash,
        response_preview=response_preview,
        validation_errors=tuple(validation_errors),
        raw_response=raw_response,
    )
    if validation_errors:
        return _judge_error(
            errors=tuple(validation_errors),
            judge_model=judge_model,
            response_hash=response_hash,
            response_preview=response_preview,
            raw_response=raw_response,
        )
    return grade


def _text(
    data: Mapping[str, object],
    key: str,
    errors: list[str],
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key}: expected non-empty string")
        return ""
    return value


def _verdict(value: str, errors: list[str]) -> EvalVerdict:
    try:
        verdict = EvalVerdict(value)
    except ValueError:
        errors.append("verdict: expected pass or fail")
        return EvalVerdict.JUDGE_ERROR
    if verdict not in {EvalVerdict.PASS, EvalVerdict.FAIL}:
        errors.append("verdict: expected pass or fail")
    return verdict


def _int_mapping(
    data: Mapping[str, object],
    key: str,
    errors: list[str],
) -> dict[str, int]:
    value = data.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}: expected object")
        return {}
    parsed: dict[str, int] = {}
    for item_key, item_value in cast("Mapping[str, object]", value).items():
        if type(item_value) is not int:
            errors.append(f"{key}.{item_key}: expected integer")
        else:
            parsed[item_key] = item_value
    return parsed


def _str_mapping(
    data: Mapping[str, object],
    key: str,
    errors: list[str],
) -> dict[str, str]:
    value = data.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}: expected object")
        return {}
    parsed: dict[str, str] = {}
    for item_key, item_value in cast("Mapping[str, object]", value).items():
        if not isinstance(item_value, str) or not item_value.strip():
            errors.append(f"{key}.{item_key}: expected non-empty string")
        else:
            parsed[item_key] = item_value
    return parsed


def _float(
    data: Mapping[str, object],
    key: str,
    errors: list[str],
) -> float:
    value = data.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        errors.append(f"{key}: expected number")
        return 0.0
    return float(value)


def _bool(
    data: Mapping[str, object],
    key: str,
    errors: list[str],
) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        errors.append(f"{key}: expected boolean")
        return True
    return value


def _judge_error(
    errors: tuple[str, ...],
    judge_model: str,
    response_hash: str,
    response_preview: str,
    raw_response: str,
) -> JudgeGrade:
    return JudgeGrade(
        verdict=EvalVerdict.JUDGE_ERROR,
        scores={},
        reasons={},
        confidence=0.0,
        ambiguous=True,
        error_message=errors[0],
        judge_model=judge_model,
        response_hash=response_hash,
        response_preview=response_preview,
        validation_errors=errors,
        raw_response=raw_response,
    )


def _judge_preview(content: str) -> str:
    without_thinking = _THINKING_PATTERN.sub(
        "[hidden reasoning omitted]",
        content,
    )
    return sanitize_text(without_thinking)
