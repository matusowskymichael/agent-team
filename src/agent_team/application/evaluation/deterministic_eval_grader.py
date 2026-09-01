"""Application service for deterministic evaluation grading."""

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from agent_team.domain.evaluation.candidate_run_result import (
    CandidateRunResult,
)
from agent_team.domain.evaluation.database_effect import DatabaseEffect
from agent_team.domain.evaluation.deterministic_grade import DeterministicGrade
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.expected_database_effect import (
    ExpectedDatabaseEffect,
)
from agent_team.domain.evaluation.expected_error import ExpectedError
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)
from agent_team.domain.evaluation.observed_tool_call import ObservedToolCall


@dataclass(frozen=True, slots=True)
class DeterministicEvalGrader:
    """Grade objective case expectations using observable data only."""

    def grade(
        self,
        case: EvalCase,
        candidate: CandidateRunResult,
        candidate_model: str,
    ) -> DeterministicGrade:
        """Grade one candidate result with deterministic hard gates."""
        reasons: list[str] = []
        hard_gate_failed = False

        if candidate.role is not case.active_role:
            reasons.append("selected role did not match the eval case")
            hard_gate_failed = True
        if candidate.model != candidate_model:
            reasons.append("selected model did not match runtime config")
            hard_gate_failed = True
        hard_gate_failed = _check_candidate_status(
            case.expected_error,
            candidate,
            reasons,
            hard_gate_failed,
        )

        hard_gate_failed = _check_forbidden_tool_calls(
            case,
            candidate.tool_calls,
            reasons,
            hard_gate_failed,
        )
        _check_required_tool_calls(case, candidate.tool_calls, reasons)
        hard_gate_failed = _check_database_effects(
            case,
            candidate.database_effects,
            reasons,
            hard_gate_failed,
        )
        _check_response_claims(
            facts=case.objective_response_facts,
            claims=case.prohibited_objective_claims,
            response=candidate.final_response,
            reasons=reasons,
        )

        return DeterministicGrade(
            passed=not reasons,
            hard_gate_failed=hard_gate_failed,
            reasons=tuple(reasons),
        )


def _check_candidate_status(
    expected_error: ExpectedError | None,
    candidate: CandidateRunResult,
    reasons: list[str],
    hard_gate_failed: bool,
) -> bool:
    if expected_error is None:
        return _check_completed_candidate(candidate, reasons, hard_gate_failed)

    error_reason = _expected_error_mismatch(expected_error, candidate)
    if error_reason is not None:
        reasons.append(error_reason)
        return True
    return hard_gate_failed


def _check_completed_candidate(
    candidate: CandidateRunResult,
    reasons: list[str],
    hard_gate_failed: bool,
) -> bool:
    if candidate.status == "completed":
        return hard_gate_failed
    reasons.append(
        f"candidate failed {candidate.error_type or 'UnknownError'}",
    )
    return True


def _expected_error_mismatch(
    expected_error: ExpectedError,
    candidate: CandidateRunResult,
) -> str | None:
    if candidate.status != "failed":
        return f"missing expected error {expected_error.error_type}"
    if candidate.error_type != expected_error.error_type:
        return f"wrong error type {candidate.error_type or 'UnknownError'}"
    if candidate.error_stage != expected_error.stage.value:
        return f"wrong error stage {candidate.error_stage or 'unknown'}"
    error_message = candidate.error_message or ""
    if (
        expected_error.message_fragment is not None
        and expected_error.message_fragment not in error_message
    ):
        return "expected error message fragment was not present"
    return None


def _check_forbidden_tool_calls(
    case: EvalCase,
    observed_calls: tuple[ObservedToolCall, ...],
    reasons: list[str],
    hard_gate_failed: bool,
) -> bool:
    observed_names = [call.name for call in observed_calls]
    forbidden_tools = set(case.forbidden_tool_calls)
    for trajectory in case.acceptable_tool_trajectories:
        forbidden_tools.update(trajectory.forbidden_tool_calls)
    for forbidden_tool in forbidden_tools:
        if forbidden_tool in observed_names:
            reasons.append(f"forbidden tool call attempted {forbidden_tool}")
            hard_gate_failed = True
        for call in observed_calls:
            if call.name == forbidden_tool and call.reached_mcp:
                reasons.append(f"forbidden tool reached MCP {forbidden_tool}")
                hard_gate_failed = True
    return hard_gate_failed


def _check_required_tool_calls(
    case: EvalCase,
    observed_calls: tuple[ObservedToolCall, ...],
    reasons: list[str],
) -> None:
    if case.acceptable_tool_trajectories:
        if not _matches_any_trajectory(
            case.acceptable_tool_trajectories,
            observed_calls,
        ):
            reasons.append("missing acceptable tool trajectory")
        return

    for expected_call in case.expected_tool_calls:
        if not _has_expected_call(expected_call, observed_calls):
            reasons.append(f"missing expected tool call {expected_call.name}")


def _check_database_effects(
    case: EvalCase,
    observed_effects: tuple[DatabaseEffect, ...],
    reasons: list[str],
    hard_gate_failed: bool,
) -> bool:
    unmatched_effects = list(observed_effects)

    for forbidden_effect in case.forbidden_database_effects:
        if _has_expected_effect(forbidden_effect, observed_effects):
            reasons.append(
                "forbidden database effect occurred "
                f"{forbidden_effect.table}.{forbidden_effect.operation}",
            )
            hard_gate_failed = True

    for expected_effect in case.expected_database_effects:
        matched_index = _matching_effect_index(
            expected_effect,
            tuple(unmatched_effects),
        )
        if matched_index is None:
            reasons.append(
                "missing expected database effect "
                f"{expected_effect.table}.{expected_effect.operation}",
            )
            hard_gate_failed = True
        else:
            del unmatched_effects[matched_index]

    for unexpected_effect in unmatched_effects:
        reasons.append(
            "unexpected database effect "
            f"{unexpected_effect.table}.{unexpected_effect.operation} "
            f"{_effect_detail(unexpected_effect)}",
        )
        hard_gate_failed = True

    return hard_gate_failed


def _check_response_claims(
    facts: tuple[str, ...],
    claims: tuple[str, ...],
    response: str,
    reasons: list[str],
) -> None:
    normalized_response = _normalize_objective_text(response)
    for fact in facts:
        if _normalize_objective_text(fact) not in normalized_response:
            reasons.append(f"missing required response fact {fact!r}")
    for claim in claims:
        normalized_claim = _normalize_objective_text(claim)
        if _contains_forbidden_claim(normalized_response, normalized_claim):
            reasons.append(f"forbidden response claim present {claim!r}")


def _has_expected_call(
    expected_call: ExpectedToolCall,
    observed_calls: tuple[ObservedToolCall, ...],
) -> bool:
    return any(
        call.name == expected_call.name
        and _contains_subset(call.arguments, expected_call.arguments_subset)
        for call in observed_calls
    )


def _matches_any_trajectory(
    trajectories: tuple[ExpectedToolTrajectory, ...],
    observed_calls: tuple[ObservedToolCall, ...],
) -> bool:
    return any(
        _matches_trajectory(trajectory, observed_calls)
        for trajectory in trajectories
    )


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


def _has_expected_effect(
    expected_effect: ExpectedDatabaseEffect,
    observed_effects: tuple[DatabaseEffect, ...],
) -> bool:
    matched_index = _matching_effect_index(expected_effect, observed_effects)
    return matched_index is not None


def _matching_effect_index(
    expected_effect: ExpectedDatabaseEffect,
    observed_effects: tuple[DatabaseEffect, ...],
) -> int | None:
    for index, effect in enumerate(observed_effects):
        if (
            effect.table == expected_effect.table
            and effect.operation == expected_effect.operation
            and _contains_subset(
                effect.field_values,
                expected_effect.field_values,
            )
        ):
            return index
    return None


def _effect_detail(effect: DatabaseEffect) -> str:
    fields = {
        key: effect.field_values[key]
        for key in _DIAGNOSTIC_FIELD_NAMES
        if key in effect.field_values
    }
    if not fields:
        fields = {
            key: effect.field_values[key]
            for key in sorted(effect.field_values)[:3]
        }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


_DIAGNOSTIC_FIELD_NAMES = (
    "id",
    "feature_id",
    "title",
    "kind",
    "assigned_role",
    "status",
    "before",
    "after",
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


def _contains_forbidden_claim(response: str, claim: str) -> bool:
    if " " in claim:
        return claim in response
    pattern = re.compile(
        rf"(?<![a-z0-9_]){re.escape(claim)}(?![a-z0-9_])",
    )
    for match in pattern.finditer(response):
        if not _is_metadata_label(response, match.end()):
            return not _is_negated_refusal(response, match.start())
    return False


def _normalize_objective_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("`", "")
    normalized = normalized.replace("*", "")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    return " ".join(normalized.split())


def _is_metadata_label(response: str, end_index: int) -> bool:
    suffix = response[end_index : end_index + 8]
    return suffix.startswith(":") or suffix.startswith(" at")


def _is_negated_refusal(response: str, start_index: int) -> bool:
    window = response[max(0, start_index - 60) : start_index]
    refusal_markers = (
        "cannot",
        "can't",
        "can not",
        "unable",
        "not allowed",
        "do not have",
        "don't have",
        "lack permission",
    )
    return any(marker in window for marker in refusal_markers)
