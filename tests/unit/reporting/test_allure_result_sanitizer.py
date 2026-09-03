"""Tests for the fail-closed Allure result sanitization boundary."""

import json
from pathlib import Path
from typing import cast

import pytest

from tests.reporting.allure_result_sanitizer import (
    AllureResultSanitizationError,
    main,
    sanitize_allure_results,
)


class TestAllureResultSanitizer:
    """Publishable result creation and failure behavior tests."""

    def test_sanitizes_failures_parameters_and_text_attachments(
        self,
        tmp_path: Path,
    ) -> None:
        """Remove every private sentinel while retaining safe diagnostics."""
        raw_directory = tmp_path / "raw"
        output_directory = tmp_path / "sanitized"
        raw_directory.mkdir()
        sentinels = {
            "status": "status-secret-4f0dd",
            "stdout": "stdout-secret-a13c2",
            "stderr": "stderr-secret-b48e1",
            "log": "log-secret-c7963",
            "json": "json-secret-d26f4",
            "text": "text-secret-e18a5",
            "parameter": "argument-secret-f31b6",
        }
        _write_raw_result(raw_directory, sentinels)

        written = sanitize_allure_results(
            raw_directory,
            output_directory,
            test_selection="not ollama and not ollama_eval",
            environment={"SECRET_TOKEN": "environment-secret-62ca7"},
        )

        assert written
        _assert_sentinels_absent(
            output_directory,
            (*sentinels.values(), "environment-secret-62ca7"),
        )
        result = _load_result(output_directory)
        details = cast("dict[str, object]", result["statusDetails"])
        assert details == {
            "message": "RuntimeError: [REDACTED]",
            "trace": ("RuntimeError: traceback omitted by reporting policy"),
        }
        parameters = cast(
            "list[dict[str, object]]",
            result["parameters"],
        )
        parameter_values = {
            str(parameter["name"]): parameter["value"]
            for parameter in parameters
        }
        assert parameter_values == {
            "arguments_json": "<redacted>",
            "role": "'business_analyst'",
        }
        diagnostic = json.loads(
            (output_directory / "diagnostic-attachment.json").read_text(
                encoding="utf-8",
            ),
        )
        assert diagnostic["phase"] == "call"
        assert diagnostic["exception_type"] == "RuntimeError"
        assert (
            output_directory / "binary-attachment.png"
        ).read_bytes() == b"\x89PNG\r\n\x1a\n\x00\xff"
        assert (output_directory / "categories.json").exists()
        assert (output_directory / "environment.properties").exists()

    def test_invalid_input_removes_stale_publishable_output(
        self,
        tmp_path: Path,
    ) -> None:
        """Leave no uploadable directory after validation fails."""
        raw_directory = tmp_path / "raw"
        output_directory = tmp_path / "sanitized"
        raw_directory.mkdir()
        output_directory.mkdir()
        (output_directory / "stale.txt").write_text(
            "must-not-survive",
            encoding="utf-8",
        )
        (raw_directory / "broken-result.json").write_text(
            "not-json private-failure-sentinel",
            encoding="utf-8",
        )

        with pytest.raises(AllureResultSanitizationError):
            sanitize_allure_results(
                raw_directory,
                output_directory,
                test_selection="unit",
                environment={},
            )

        assert not output_directory.exists()

    def test_missing_attachment_fails_closed(self, tmp_path: Path) -> None:
        """Reject a result whose declared attachment cannot be sanitized."""
        raw_directory = tmp_path / "raw"
        raw_directory.mkdir()
        _write_json(
            raw_directory / "missing-result.json",
            {
                "name": "Missing attachment",
                "status": "failed",
                "attachments": [
                    {
                        "name": "stdout",
                        "source": "missing-attachment.txt",
                        "type": "text/plain",
                    },
                ],
            },
        )

        with pytest.raises(AllureResultSanitizationError):
            sanitize_allure_results(
                raw_directory,
                tmp_path / "sanitized",
                test_selection="unit",
                environment={},
            )

        assert not (tmp_path / "sanitized").exists()

    def test_cli_reports_failure_without_private_details(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Return a concise error without printing raw result contents."""
        raw_directory = tmp_path / "raw"
        raw_directory.mkdir()
        sentinel = "cli-sanitization-secret-921ad"
        (raw_directory / "broken-result.json").write_text(
            f"invalid {sentinel}",
            encoding="utf-8",
        )

        exit_code = main(
            [
                "--source-dir",
                str(raw_directory),
                "--output-dir",
                str(tmp_path / "sanitized"),
            ],
        )

        output = capsys.readouterr()
        assert exit_code == 1
        assert output.out == ""
        assert output.err == "Allure result sanitization failed.\n"
        assert sentinel not in output.err


def _write_raw_result(
    directory: Path,
    sentinels: dict[str, str],
) -> None:
    attachments = [
        ("stdout", "stdout-attachment.txt", "text/plain"),
        ("stderr", "stderr-attachment.txt", "text/plain"),
        ("log", "log-attachment.txt", "text/plain"),
        ("Tool payload", "payload-attachment.json", "application/json"),
        ("Notes", "notes-attachment.txt", "text/plain"),
        (
            "Sanitized failure diagnostic",
            "diagnostic-attachment.json",
            "application/json",
        ),
        ("Screenshot", "binary-attachment.png", "image/png"),
    ]
    _write_json(
        directory / "failure-result.json",
        {
            "name": f"Failure [prompt={sentinels['parameter']}]",
            "status": "failed",
            "statusDetails": {
                "message": f"RuntimeError: password={sentinels['status']}",
                "trace": (
                    f"raise RuntimeError('password={sentinels['status']}')"
                ),
            },
            "parameters": [
                {
                    "name": "arguments_json",
                    "value": sentinels["parameter"],
                },
                {"name": "role", "value": "'business_analyst'"},
            ],
            "attachments": [
                {"name": name, "source": source, "type": mime_type}
                for name, source, mime_type in attachments
            ],
            "uuid": "safe-result-id",
        },
    )
    for name in ("stdout", "stderr", "log"):
        (directory / f"{name}-attachment.txt").write_text(
            sentinels[name],
            encoding="utf-8",
        )
    _write_json(
        directory / "payload-attachment.json",
        {"arguments_json": sentinels["json"]},
    )
    (directory / "notes-attachment.txt").write_text(
        sentinels["text"],
        encoding="utf-8",
    )
    _write_json(
        directory / "diagnostic-attachment.json",
        {
            "node_id": "tests/unit/test_sample.py::TestSample::test_failure",
            "phase": "call",
            "markers": ["security"],
            "duration_seconds": 0.1,
            "exception_type": "RuntimeError",
            "python_version": "3.14.7",
            "platform": "Linux x86_64",
            "unexpected": sentinels["json"],
        },
    )
    (directory / "binary-attachment.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\xff",
    )


def _load_result(directory: Path) -> dict[str, object]:
    loaded = cast(
        object,
        json.loads(
            next(directory.glob("*-result.json")).read_text(
                encoding="utf-8",
            ),
        ),
    )
    assert isinstance(loaded, dict)
    return cast("dict[str, object]", loaded)


def _assert_sentinels_absent(
    directory: Path,
    sentinels: tuple[str, ...],
) -> None:
    for path in directory.iterdir():
        content = path.read_bytes()
        for sentinel in sentinels:
            assert sentinel.encode() not in content


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
