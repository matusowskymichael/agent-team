"""JSONL golden dataset loader."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from agent_team.application.evaluation.golden_dataset_loader import (
    GoldenDatasetLoader,
)
from agent_team.domain.evaluation.eval_artifact_fixture import (
    EvalArtifactFixture,
)
from agent_team.domain.evaluation.eval_case import EvalCase
from agent_team.domain.evaluation.eval_case_intent import EvalCaseIntent
from agent_team.domain.evaluation.eval_context_policy import (
    EvalContextPolicy,
)
from agent_team.domain.evaluation.eval_error_stage import EvalErrorStage
from agent_team.domain.evaluation.eval_feature_fixture import (
    EvalFeatureFixture,
)
from agent_team.domain.evaluation.eval_session_fixture import (
    EvalSessionFixture,
)
from agent_team.domain.evaluation.eval_suite import EvalSuite
from agent_team.domain.evaluation.eval_task_fixture import EvalTaskFixture
from agent_team.domain.evaluation.eval_workspace_file_fixture import (
    EvalWorkspaceFileFixture,
)
from agent_team.domain.evaluation.expected_database_effect import (
    ExpectedDatabaseEffect,
)
from agent_team.domain.evaluation.expected_error import ExpectedError
from agent_team.domain.evaluation.expected_tool_call import ExpectedToolCall
from agent_team.domain.evaluation.expected_tool_trajectory import (
    ExpectedToolTrajectory,
)
from agent_team.domain.runtime.development_role import DevelopmentRole
from agent_team.domain.workflow.artifact_kind import ArtifactKind
from agent_team.domain.workflow.feature_status import FeatureStatus
from agent_team.domain.workflow.task_status import TaskStatus
from agent_team.infrastructure.evaluation.eval_hashes import hash_file


@dataclass(frozen=True, slots=True)
class JsonlGoldenDatasetLoader:
    """Load manually maintained JSONL golden datasets."""

    validator: GoldenDatasetLoader = field(
        default_factory=GoldenDatasetLoader,
    )

    def load(self, suite_id: str, path: Path) -> EvalSuite:
        """Load and validate a JSONL evaluation suite."""
        metadata, case_lines = _records(path)
        cases = tuple(
            _parse_case(line, path, line_number)
            for line_number, line in case_lines
        )
        suite = self.validator.build_suite(
            suite_id=suite_id,
            dataset_hash=hash_file(path),
            cases=cases,
        )
        return EvalSuite(
            id=suite.id,
            cases=suite.cases,
            dataset_hash=suite.dataset_hash,
            dataset_version=metadata.get("dataset_version"),
        )


def _records(
    path: Path,
) -> tuple[dict[str, str], tuple[tuple[int, str], ...]]:
    metadata: dict[str, str] = {}
    cases: list[tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parsed = _json_object(line, path, line_number)
        if parsed.get("record_type") == "dataset_metadata":
            metadata = _metadata(parsed)
        else:
            cases.append((line_number, line))
    return metadata, tuple(cases)


def _parse_case(line: str, path: Path, line_number: int) -> EvalCase:
    data = _json_object(line, path, line_number)
    objective_facts = _texts(
        data.get(
            "objective_response_facts",
            data.get("required_response_facts", []),
        ),
    )
    prohibited_claims = _texts(
        data.get(
            "prohibited_objective_claims",
            data.get("forbidden_response_claims", []),
        ),
    )
    return EvalCase(
        id=_text(data, "id"),
        name=_text(data, "name"),
        category=_text(data, "category"),
        severity=_text(data, "severity"),
        active_role=DevelopmentRole(_text(data, "active_role")),
        feature_fixtures=_features(data.get("feature_fixtures", [])),
        session_fixtures=_sessions(data.get("session_fixtures", [])),
        prior_session_turns=_texts(data.get("prior_session_turns", [])),
        user_input=_text(data, "user_input"),
        expected_tool_calls=_expected_tool_calls(
            data.get("expected_tool_calls", []),
        ),
        forbidden_tool_calls=_texts(data.get("forbidden_tool_calls", [])),
        expected_database_effects=_database_effects(
            data.get("expected_database_effects", []),
        ),
        forbidden_database_effects=_database_effects(
            data.get("forbidden_database_effects", []),
        ),
        required_response_facts=_texts(
            data.get("required_response_facts", []),
        ),
        forbidden_response_claims=_texts(
            data.get("forbidden_response_claims", []),
        ),
        rubric_id=_text(data, "rubric_id"),
        note=_text(data, "note"),
        objective_response_facts=objective_facts,
        semantic_response_requirements=_texts(
            data.get("semantic_response_requirements", []),
        ),
        prohibited_objective_claims=prohibited_claims,
        acceptable_tool_trajectories=_tool_trajectories(
            data.get("acceptable_tool_trajectories", []),
        ),
        expected_error=_expected_error(data),
        semantic_judge_required=_optional_bool(
            data.get("semantic_judge_required"),
            default=True,
        ),
        intent=_case_intent(data.get("intent")),
        context_policy=_context_policy(data.get("context_policy")),
        requested_session_id=_optional_text(data.get("requested_session_id")),
        feature_scope_id=_optional_int(data.get("feature_scope_id")),
        task_scope_id=_optional_int(data.get("task_scope_id")),
        workspace_files=_workspace_files(data.get("workspace_files", [])),
        adjudication_note=_optional_text(data.get("adjudication_note")) or "",
        max_output_tokens=_optional_int(data.get("max_output_tokens")),
    )


def _json_object(
    line: str,
    path: Path,
    line_number: int,
) -> Mapping[str, object]:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path}:{line_number} contains invalid JSON.",
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}:{line_number} must contain a JSON object.")
    return cast("Mapping[str, object]", parsed)


def _metadata(data: Mapping[str, object]) -> dict[str, str]:
    dataset_version = _optional_text(data.get("dataset_version"))
    if dataset_version is None:
        return {}
    return {"dataset_version": dataset_version}


def _features(value: object) -> tuple[EvalFeatureFixture, ...]:
    return tuple(_feature(item) for item in _objects(value))


def _feature(value: Mapping[str, object]) -> EvalFeatureFixture:
    return EvalFeatureFixture(
        id=_int(value, "id"),
        title=_text(value, "title"),
        description=_text(value, "description"),
        status=FeatureStatus(_text(value, "status")),
        artifacts=_artifacts(value.get("artifacts", [])),
        tasks=_tasks(value.get("tasks", [])),
    )


def _artifacts(value: object) -> tuple[EvalArtifactFixture, ...]:
    return tuple(
        EvalArtifactFixture(
            feature_id=_int(item, "feature_id"),
            kind=ArtifactKind(_text(item, "kind")),
            content=_text(item, "content"),
            created_by=_text(item, "created_by"),
        )
        for item in _objects(value)
    )


def _tasks(value: object) -> tuple[EvalTaskFixture, ...]:
    return tuple(
        EvalTaskFixture(
            feature_id=_int(item, "feature_id"),
            title=_text(item, "title"),
            description=_text(item, "description"),
            assigned_role=DevelopmentRole(_text(item, "assigned_role")),
            status=TaskStatus(_text(item, "status")),
        )
        for item in _objects(value)
    )


def _sessions(value: object) -> tuple[EvalSessionFixture, ...]:
    return tuple(
        EvalSessionFixture(
            session_id=_text(item, "session_id"),
            feature_id=_int(item, "feature_id"),
            role=DevelopmentRole(_text(item, "role")),
        )
        for item in _objects(value)
    )


def _workspace_files(value: object) -> tuple[EvalWorkspaceFileFixture, ...]:
    return tuple(
        EvalWorkspaceFileFixture(
            path=_text(item, "path"),
            content=_text(item, "content"),
        )
        for item in _objects(value)
    )


def _expected_tool_calls(value: object) -> tuple[ExpectedToolCall, ...]:
    return tuple(
        ExpectedToolCall(
            name=_text(item, "name"),
            arguments_subset=_object_dict(
                item.get("arguments_subset", {}),
            ),
            order=_optional_int(item.get("order")),
        )
        for item in _objects(value)
    )


def _tool_trajectories(
    value: object,
) -> tuple[ExpectedToolTrajectory, ...]:
    return tuple(
        ExpectedToolTrajectory(
            required_tool_calls=_expected_tool_calls(
                item.get("required_tool_calls", []),
            ),
            order_matters=_optional_bool(
                item.get("order_matters"),
                default=False,
            ),
            optional_read_only_tool_calls=_texts(
                item.get("optional_read_only_tool_calls", []),
            ),
            forbidden_tool_calls=_texts(item.get("forbidden_tool_calls", [])),
        )
        for item in _objects(value)
    )


def _expected_error(value: Mapping[str, object]) -> ExpectedError | None:
    if "expected_error" in value:
        data = _object(value.get("expected_error"))
    elif "expected_error_type" in value:
        data = value
    else:
        return None
    message_fragment = _optional_text(
        data.get(
            "expected_error_message_fragment",
            data.get("expected_error_message_code"),
        ),
    )
    return ExpectedError(
        error_type=_text(data, "expected_error_type"),
        stage=EvalErrorStage(_text(data, "expected_error_stage")),
        message_fragment=message_fragment,
    )


def _case_intent(value: object) -> EvalCaseIntent:
    if value is None:
        return EvalCaseIntent.UNSPECIFIED
    return EvalCaseIntent(_string(value))


def _context_policy(value: object) -> EvalContextPolicy:
    if value is None:
        return EvalContextPolicy.STANDARD_FEATURE_CONTEXT
    return EvalContextPolicy(_string(value))


def _database_effects(value: object) -> tuple[ExpectedDatabaseEffect, ...]:
    return tuple(
        ExpectedDatabaseEffect(
            table=_text(item, "table"),
            operation=_text(item, "operation"),
            field_values=_object_dict(item.get("field_values", {})),
        )
        for item in _objects(value)
    )


def _objects(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of objects.")
    values = cast("list[object]", value)
    objects: list[Mapping[str, object]] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("Expected a JSON object.")
        objects.append(cast("Mapping[str, object]", item))
    return tuple(objects)


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return cast("Mapping[str, object]", value)


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings.")
    values = cast("list[object]", value)
    return tuple(_string(item) for item in values)


def _text(data: Mapping[str, object], key: str) -> str:
    return _string(data.get(key))


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string value.")
    return value


def _int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError("Expected an integer value.")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("Expected an integer value.")
    return value


def _optional_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("Expected a boolean value.")
    return value


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected an object value.")
    return dict(cast("Mapping[str, object]", value))
