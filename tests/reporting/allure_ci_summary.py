"""Write a concise GitHub summary from raw Allure test results."""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

_STATUSES = ("passed", "failed", "broken", "skipped", "unknown")
_STEP_STATUSES = ("success", "failure", "skipped", "cancelled")


def result_counts(results_directory: Path) -> dict[str, int]:
    """Count Allure result statuses without exposing test content."""
    counts: Counter[str] = Counter()
    for result_path in results_directory.glob("*-result.json"):
        try:
            loaded = cast(
                object,
                json.loads(result_path.read_text(encoding="utf-8")),
            )
        except json.JSONDecodeError, OSError:
            counts["unknown"] += 1
            continue
        mapping = (
            cast("dict[object, object]", loaded)
            if isinstance(loaded, dict)
            else {}
        )
        status = mapping.get("status")
        counts[status if status in _STATUSES else "unknown"] += 1
    return {status: counts[status] for status in _STATUSES}


def github_summary(
    counts: dict[str, int],
    *,
    pytest_status: str,
    report_status: str,
    agent_status: str,
    quality_gate_status: str,
) -> str:
    """Render safe test totals, generation states, and artifact names."""
    total = sum(counts.values())
    lines = [
        "## Allure Report",
        "",
        f"Tests: **{total}** total",
        "",
        "| Passed | Failed | Broken | Skipped | Unknown |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {counts['passed']} | {counts['failed']} | "
            f"{counts['broken']} | {counts['skipped']} | "
            f"{counts['unknown']} |"
        ),
        "",
        f"- pytest: `{pytest_status}`",
        f"- HTML report: `{report_status}`",
        f"- Agent report: `{agent_status}`",
        f"- Quality gate: `{quality_gate_status}`",
        "- Artifacts: `allure-results`, `allure-report`, "
        "`allure-agent-report`",
        "",
    ]
    return "\n".join(lines)


def main(arguments: Sequence[str] | None = None) -> int:
    """Append the Allure summary to a requested GitHub summary file."""
    parser = argparse.ArgumentParser(
        description="Create the Agent Team Allure CI summary.",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pytest-status",
        choices=_STEP_STATUSES,
        required=True,
    )
    parser.add_argument(
        "--report-status",
        choices=_STEP_STATUSES,
        required=True,
    )
    parser.add_argument(
        "--agent-status",
        choices=_STEP_STATUSES,
        required=True,
    )
    parser.add_argument(
        "--quality-gate-status",
        choices=_STEP_STATUSES,
        required=True,
    )
    parsed = parser.parse_args(arguments)
    summary = github_summary(
        result_counts(parsed.results_dir),
        pytest_status=parsed.pytest_status,
        report_status=parsed.report_status,
        agent_status=parsed.agent_status,
        quality_gate_status=parsed.quality_gate_status,
    )
    with parsed.output.open("a", encoding="utf-8") as stream:
        stream.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
