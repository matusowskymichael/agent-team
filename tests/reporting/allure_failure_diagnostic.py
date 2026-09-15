"""Create deliberately narrow diagnostics for failed pytest phases."""

import json
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass

from tests.reporting.allure_ci_context import (
    commit_sha_from,
    run_id_from,
)
from tests.reporting.allure_test_policy import safe_node_id


@dataclass(frozen=True, slots=True)
class FailureDiagnosticContext:
    """Safe execution facts for one failed pytest phase."""

    node_id: str
    phase: str
    marker_names: tuple[str, ...]
    duration_seconds: float
    exception_type: str


def failure_diagnostic_json(
    context: FailureDiagnosticContext,
    environment: Mapping[str, str],
) -> str:
    """Serialize a sanitized failure diagnostic without exception details."""
    diagnostic: dict[str, object] = {
        "node_id": safe_node_id(context.node_id),
        "phase": context.phase,
        "markers": sorted(context.marker_names),
        "duration_seconds": round(max(context.duration_seconds, 0.0), 6),
        "exception_type": _safe_exception_type(context.exception_type),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
    }
    run_id = run_id_from(environment)
    commit_sha = commit_sha_from(environment)
    if run_id is not None:
        diagnostic["github_run_id"] = run_id
    if commit_sha is not None:
        diagnostic["commit_sha"] = commit_sha
    return json.dumps(diagnostic, indent=2, sort_keys=True)


def _safe_exception_type(value: str) -> str:
    candidate = value.split(":", maxsplit=1)[0].strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,119}", candidate):
        return "Exception"
    return candidate
