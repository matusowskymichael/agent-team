"""JSON evaluation result repository."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.database_effect import DatabaseEffect
from agent_team.domain.evaluation.deterministic_grade import DeterministicGrade
from agent_team.domain.evaluation.eval_attempt_result import (
    EvalAttemptResult,
)
from agent_team.domain.evaluation.eval_case_intent import EvalCaseIntent
from agent_team.domain.evaluation.eval_case_result import EvalCaseResult
from agent_team.domain.evaluation.eval_context_policy import (
    EvalContextPolicy,
)
from agent_team.domain.evaluation.eval_run_result import EvalRunResult
from agent_team.domain.evaluation.eval_verdict import EvalVerdict
from agent_team.domain.evaluation.judge_grade import JudgeGrade
from agent_team.domain.evaluation.observed_skill_call import ObservedSkillCall
from agent_team.domain.evaluation.observed_tool_call import ObservedToolCall
from agent_team.domain.runtime.development_role import DevelopmentRole

DEFAULT_EVAL_RESULTS_DIR = Path(".agent_team/evals")


@dataclass(frozen=True, slots=True)
class JsonEvalResultRepository:
    """Persist evaluation results as local JSON files."""

    directory: Path = DEFAULT_EVAL_RESULTS_DIR

    def __post_init__(self) -> None:
        """Create the local result directory."""
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: EvalRunResult) -> None:
        """Persist an evaluation run result without modifying inputs."""
        path = self.directory / f"{result.id}.json"
        path.write_text(
            json.dumps(_run_to_dict(result), indent=2, sort_keys=True),
        )

    def get(self, result_id: str) -> EvalRunResult | None:
        """Return a saved evaluation result, if it exists."""
        path = self.directory / f"{result_id}.json"
        if not path.exists():
            return None
        parsed = json.loads(path.read_text())
        if not isinstance(parsed, dict):
            raise ValueError("Eval result JSON must contain an object.")
        return _run_from_mapping(cast("Mapping[str, object]", parsed))

    def list_ids(self) -> list[str]:
        """Return saved evaluation run IDs."""
        return sorted(path.stem for path in self.directory.glob("*.json"))


def _run_to_dict(result: EvalRunResult) -> dict[str, object]:
    return {
        "id": result.id,
        "suite_id": result.suite_id,
        "candidate_model": result.candidate_model,
        "judge_model": result.judge_model,
        "dataset_hash": result.dataset_hash,
        "rubric_hash": result.rubric_hash,
        "instructions_hash": result.instructions_hash,
        "package_version": result.package_version,
        "started_at": result.started_at.isoformat(),
        "ended_at": result.ended_at.isoformat(),
        "duration_seconds": result.duration_seconds,
        "warnings": list(result.warnings),
        "case_filter": result.case_filter,
        "candidate_thinking_enabled": result.candidate_thinking_enabled,
        "judge_thinking_enabled": result.judge_thinking_enabled,
        "case_results": [
            _case_result_to_dict(case_result)
            for case_result in result.case_results
        ],
    }


def _case_result_to_dict(result: EvalCaseResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "repetition": result.repetition,
        "candidate_result": _candidate_to_dict(result.candidate_result),
        "deterministic_grade": _deterministic_to_dict(
            result.deterministic_grade,
        ),
        "judge_grade": None
        if result.judge_grade is None
        else _judge_to_dict(result.judge_grade),
        "verdict": result.verdict.value,
        "semantic_judge_required": result.semantic_judge_required,
        "intent": result.intent.value,
        "context_policy": result.context_policy.value,
        "candidate_duration_seconds": result.candidate_duration_seconds,
        "deterministic_duration_seconds": (
            result.deterministic_duration_seconds
        ),
        "judge_duration_seconds": result.judge_duration_seconds,
        "total_duration_seconds": result.total_duration_seconds,
    }


def _candidate_to_dict(result: CandidateRunResult) -> dict[str, object]:
    return {
        "role": result.role.value,
        "model": result.model,
        "final_response": result.final_response,
        "tool_calls": [_tool_call_to_dict(call) for call in result.tool_calls],
        "skill_calls": [
            _skill_call_to_dict(call) for call in result.skill_calls
        ],
        "database_effects": [
            _database_effect_to_dict(effect)
            for effect in result.database_effects
        ],
        "status": result.status,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "error_stage": result.error_stage,
        "attempt_count": result.attempt_count,
        "retry_count": result.retry_count,
        "attempts": [_attempt_to_dict(attempt) for attempt in result.attempts],
        "max_output_tokens": result.max_output_tokens,
    }


def _tool_call_to_dict(call: ObservedToolCall) -> dict[str, object]:
    return {
        "name": call.name,
        "arguments": call.arguments,
        "status": call.status,
        "reached_mcp": call.reached_mcp,
    }


def _skill_call_to_dict(call: ObservedSkillCall) -> dict[str, object]:
    return {
        "tool_name": call.tool_name,
        "skill_name": call.skill_name,
        "status": call.status,
        "content_hash": call.content_hash,
        "resource_name": call.resource_name,
    }


def _database_effect_to_dict(effect: DatabaseEffect) -> dict[str, object]:
    return {
        "table": effect.table,
        "operation": effect.operation,
        "field_values": effect.field_values,
    }


def _attempt_to_dict(attempt: EvalAttemptResult) -> dict[str, object]:
    return {
        "attempt": attempt.attempt,
        "status": attempt.status,
        "duration_seconds": attempt.duration_seconds,
        "error_type": attempt.error_type,
        "error_stage": attempt.error_stage,
    }


def _deterministic_to_dict(grade: DeterministicGrade) -> dict[str, object]:
    return {
        "passed": grade.passed,
        "hard_gate_failed": grade.hard_gate_failed,
        "reasons": list(grade.reasons),
    }


def _judge_to_dict(grade: JudgeGrade) -> dict[str, object]:
    return {
        "verdict": grade.verdict.value,
        "scores": grade.scores,
        "reasons": grade.reasons,
        "confidence": grade.confidence,
        "ambiguous": grade.ambiguous,
        "error_message": grade.error_message,
        "rubric_id": grade.rubric_id,
        "rubric_version": grade.rubric_version,
        "case_id": grade.case_id,
        "evidence": grade.evidence,
        "judge_model": grade.judge_model,
        "response_hash": grade.response_hash,
        "response_preview": grade.response_preview,
        "validation_errors": list(grade.validation_errors),
        "retry_count": grade.retry_count,
    }


def _run_from_mapping(data: Mapping[str, object]) -> EvalRunResult:
    return EvalRunResult(
        id=_text(data, "id"),
        suite_id=_text(data, "suite_id"),
        candidate_model=_text(data, "candidate_model"),
        judge_model=_optional_text(data.get("judge_model")),
        dataset_hash=_text(data, "dataset_hash"),
        rubric_hash=_text(data, "rubric_hash"),
        instructions_hash=_text(data, "instructions_hash"),
        package_version=_text(data, "package_version"),
        started_at=datetime.fromisoformat(_text(data, "started_at")),
        ended_at=datetime.fromisoformat(_text(data, "ended_at")),
        case_results=tuple(
            _case_result_from_mapping(item)
            for item in _objects(data.get("case_results", []))
        ),
        warnings=_texts(data.get("warnings", [])),
        case_filter=_optional_text(data.get("case_filter")),
        duration_seconds=_optional_float(data.get("duration_seconds")),
        candidate_thinking_enabled=_optional_bool(
            data.get("candidate_thinking_enabled"),
            default=False,
        ),
        judge_thinking_enabled=_optional_bool_or_none(
            data.get("judge_thinking_enabled"),
        ),
    )


def _case_result_from_mapping(
    data: Mapping[str, object],
) -> EvalCaseResult:
    return EvalCaseResult(
        case_id=_text(data, "case_id"),
        repetition=_int(data, "repetition"),
        candidate_result=_candidate_from_mapping(
            _object(data.get("candidate_result")),
        ),
        deterministic_grade=_deterministic_from_mapping(
            _object(data.get("deterministic_grade")),
        ),
        judge_grade=_optional_judge(data.get("judge_grade")),
        verdict=EvalVerdict(_text(data, "verdict")),
        semantic_judge_required=_optional_bool(
            data.get("semantic_judge_required"),
            default=True,
        ),
        intent=EvalCaseIntent(
            _optional_text(data.get("intent")) or EvalCaseIntent.UNSPECIFIED,
        ),
        context_policy=EvalContextPolicy(
            _optional_text(data.get("context_policy"))
            or EvalContextPolicy.STANDARD_FEATURE_CONTEXT,
        ),
        candidate_duration_seconds=_optional_float(
            data.get("candidate_duration_seconds"),
        ),
        deterministic_duration_seconds=_optional_float(
            data.get("deterministic_duration_seconds"),
        ),
        judge_duration_seconds=_optional_float(
            data.get("judge_duration_seconds"),
        ),
        total_duration_seconds=_optional_float(
            data.get("total_duration_seconds"),
        ),
    )


def _candidate_from_mapping(
    data: Mapping[str, object],
) -> CandidateRunResult:
    return CandidateRunResult(
        role=DevelopmentRole(_text(data, "role")),
        model=_text(data, "model"),
        final_response=_text(data, "final_response"),
        tool_calls=tuple(
            _tool_call_from_mapping(item)
            for item in _objects(data.get("tool_calls", []))
        ),
        skill_calls=tuple(
            _skill_call_from_mapping(item)
            for item in _objects(data.get("skill_calls", []))
        ),
        database_effects=tuple(
            _database_effect_from_mapping(item)
            for item in _objects(data.get("database_effects", []))
        ),
        status=_text(data, "status"),
        error_type=_optional_text(data.get("error_type")),
        error_message=_optional_text(data.get("error_message")),
        error_stage=_optional_text(data.get("error_stage")),
        attempt_count=_optional_int(data.get("attempt_count"), default=1),
        retry_count=_optional_int(data.get("retry_count"), default=0),
        attempts=tuple(
            _attempt_from_mapping(item)
            for item in _objects(data.get("attempts", []))
        ),
        max_output_tokens=_optional_int_or_none(
            data.get("max_output_tokens"),
        ),
    )


def _tool_call_from_mapping(
    data: Mapping[str, object],
) -> ObservedToolCall:
    return ObservedToolCall(
        name=_text(data, "name"),
        arguments=_dict(data.get("arguments")),
        status=_text(data, "status"),
        reached_mcp=_bool(data, "reached_mcp"),
    )


def _skill_call_from_mapping(
    data: Mapping[str, object],
) -> ObservedSkillCall:
    return ObservedSkillCall(
        tool_name=_text(data, "tool_name"),
        skill_name=_text(data, "skill_name"),
        status=_text(data, "status"),
        content_hash=_optional_text(data.get("content_hash")),
        resource_name=_optional_text(data.get("resource_name")),
    )


def _database_effect_from_mapping(
    data: Mapping[str, object],
) -> DatabaseEffect:
    return DatabaseEffect(
        table=_text(data, "table"),
        operation=_text(data, "operation"),
        field_values=_dict(data.get("field_values")),
    )


def _attempt_from_mapping(
    data: Mapping[str, object],
) -> EvalAttemptResult:
    return EvalAttemptResult(
        attempt=_int(data, "attempt"),
        status=_text(data, "status"),
        duration_seconds=_optional_float(data.get("duration_seconds")),
        error_type=_optional_text(data.get("error_type")),
        error_stage=_optional_text(data.get("error_stage")),
    )


def _deterministic_from_mapping(
    data: Mapping[str, object],
) -> DeterministicGrade:
    return DeterministicGrade(
        passed=_bool(data, "passed"),
        hard_gate_failed=_bool(data, "hard_gate_failed"),
        reasons=_texts(data.get("reasons", [])),
    )


def _optional_judge(value: object) -> JudgeGrade | None:
    if value is None:
        return None
    data = _object(value)
    return JudgeGrade(
        verdict=EvalVerdict(_text(data, "verdict")),
        scores=_int_dict(data.get("scores")),
        reasons=_str_dict(data.get("reasons")),
        confidence=_float(data, "confidence"),
        ambiguous=_bool(data, "ambiguous"),
        error_message=_optional_text(data.get("error_message")),
        rubric_id=_optional_text(data.get("rubric_id")),
        rubric_version=_optional_text(data.get("rubric_version")),
        case_id=_optional_text(data.get("case_id")),
        evidence=_optional_str_dict(data.get("evidence")),
        judge_model=_optional_text(data.get("judge_model")),
        response_hash=_optional_text(data.get("response_hash")),
        response_preview=_optional_text(data.get("response_preview")),
        validation_errors=_optional_texts(data.get("validation_errors")),
        retry_count=_optional_int(data.get("retry_count")),
    )


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object.")
    return cast("Mapping[str, object]", value)


def _objects(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError("Expected JSON object list.")
    values = cast("list[object]", value)
    return tuple(_object(item) for item in values)


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object.")
    return dict(cast("Mapping[str, object]", value))


def _str_dict(value: object) -> dict[str, str]:
    return {key: _string(item) for key, item in _dict(value).items()}


def _optional_str_dict(value: object) -> dict[str, str]:
    if value is None:
        return {}
    return _str_dict(value)


def _int_dict(value: object) -> dict[str, int]:
    return {key: _int_value(item) for key, item in _dict(value).items()}


def _text(data: Mapping[str, object], key: str) -> str:
    return _string(data.get(key))


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected string.")
    return value


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("Expected string list.")
    values = cast("list[object]", value)
    return tuple(_string(item) for item in values)


def _optional_texts(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    return _texts(value)


def _int(data: Mapping[str, object], key: str) -> int:
    return _int_value(data.get(key))


def _int_value(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("Expected integer.")
    return value


def _optional_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    return _int_value(value)


def _optional_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return _int_value(value)


def _optional_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("Expected boolean.")
    return value


def _optional_bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Expected boolean.")
    return value


def _float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float):
        raise ValueError("Expected number.")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError("Expected number.")
    return float(value)


def _bool(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError("Expected boolean.")
    return value
