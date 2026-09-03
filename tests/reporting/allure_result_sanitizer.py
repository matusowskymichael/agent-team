"""Create publishable Allure results from private raw pytest output."""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

from agent_team.application.audit.audit_sanitizer import (
    REDACTED_VALUE,
    sanitize_full_text,
)
from tests.reporting.allure_report_metadata import (
    write_allure_report_metadata,
)
from tests.reporting.allure_test_policy import (
    safe_reported_parameter_value,
)

_RESULT_SUFFIXES = ("-result.json", "-container.json")
_PRIVATE_METADATA_FILES = {
    "categories.json",
    "environment.properties",
    "executor.json",
}
_ATTACHMENT_MARKER = "-attachment."
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9_.-]{1,200}")
_SAFE_MIME_TYPE = re.compile(r"[a-z0-9.+-]+/[a-z0-9.+-]+")
_SAFE_EXCEPTION_TYPE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.]{0,100}(?:Error|Exception|Failure)",
)
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,159}")
_SAFE_SHA = re.compile(r"[0-9a-f]{40}")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:arguments?(?:_json)?|authorization|content|credential|"
    r"description|message|output|password|payload|prompt|request|response|"
    r"secret|token|tool[_ -]?input|api[_ -]?key)\b\s*[:=]\s*)[^\r\n]*",
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+\S+")
_HOSTED_TOKEN = re.compile(
    r"\b(?:github_pat_|gh[opusr]_)[A-Za-z0-9_]{12,}\b",
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?"
    r"-----END [^-\r\n]*PRIVATE KEY-----",
    re.DOTALL,
)
_LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:/home/|/Users/|[A-Za-z]:\\Users\\)"
    r"[^\s\"']+",
)
_TEXT_EXTENSIONS = {
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".log",
    ".md",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_BINARY_MIME_PREFIXES = ("audio/", "font/", "image/", "video/")
_BINARY_MIME_TYPES = {
    "application/octet-stream",
    "application/pdf",
    "application/zip",
}
_DIAGNOSTIC_ATTACHMENT_NAME = "Sanitized failure diagnostic"
_TEXT_ATTACHMENT_CONTENT = (
    "[REDACTED: textual attachment omitted by reporting policy]\n"
)


class AllureResultSanitizationError(RuntimeError):
    """Raised when raw Allure results cannot be made safe to publish."""


def sanitize_allure_results(
    source_directory: Path,
    output_directory: Path,
    *,
    test_selection: str,
    environment: Mapping[str, str],
) -> tuple[Path, ...]:
    """Atomically create a validated, sanitized Allure result directory."""
    source = source_directory.resolve()
    output = output_directory.resolve()
    _validate_directories(source, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _remove_output(output)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.sanitizing-",
            dir=output.parent,
        ),
    )
    try:
        relative_paths = _sanitize_result_files(source, temporary)
        metadata_paths = write_allure_report_metadata(
            temporary,
            test_selection=test_selection,
            environment=environment,
        )
        relative_paths.extend(
            path.relative_to(temporary) for path in metadata_paths
        )
        temporary.replace(output)
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        _remove_output(output)
        if isinstance(error, AllureResultSanitizationError):
            raise
        message = "Allure result sanitization failed."
        raise AllureResultSanitizationError(message) from error
    return tuple(output / path for path in relative_paths)


def main(arguments: Sequence[str] | None = None) -> int:
    """Sanitize raw Allure output without printing private failure details."""
    parser = argparse.ArgumentParser(
        description="Create publishable Agent Team Allure results.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("allure-results-raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("allure-results"),
    )
    parser.add_argument(
        "--selection",
        default="all configured tests",
    )
    parsed = parser.parse_args(arguments)
    try:
        sanitize_allure_results(
            parsed.source_dir,
            parsed.output_dir,
            test_selection=parsed.selection,
            environment=os.environ,
        )
    except AllureResultSanitizationError:
        print("Allure result sanitization failed.", file=sys.stderr)
        return 1
    return 0


def _validate_directories(source: Path, output: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        _fail()
    if source == output or source.is_relative_to(output):
        _fail()
    if output == Path.cwd().resolve() or output.parent == output:
        _fail()


def _sanitize_result_files(source: Path, output: Path) -> list[Path]:
    documents: list[tuple[Path, dict[str, object]]] = []
    attachment_paths: set[str] = set()
    source_attachments: set[str] = set()
    for source_path in sorted(source.iterdir()):
        if not source_path.is_file() or source_path.is_symlink():
            _fail()
        if source_path.name.endswith(_RESULT_SUFFIXES):
            documents.append((source_path, _load_object(source_path)))
        elif source_path.name in _PRIVATE_METADATA_FILES:
            continue
        elif _ATTACHMENT_MARKER in source_path.name:
            source_attachments.add(source_path.name)
        else:
            _fail()
    if not any(
        path.name.endswith("-result.json") for path, _data in documents
    ):
        _fail()

    written: list[Path] = []
    attachment_descriptors: dict[str, tuple[str, str]] = {}
    for source_path, document in documents:
        sanitized = _sanitize_value(document, attachment_descriptors)
        relative_path = Path(source_path.name)
        _write_json(output / relative_path, sanitized)
        written.append(relative_path)
    attachment_paths.update(attachment_descriptors)
    if source_attachments != attachment_paths:
        _fail()
    for source_name, (attachment_name, mime_type) in sorted(
        attachment_descriptors.items(),
    ):
        source_path = source / source_name
        output_path = output / source_name
        _sanitize_attachment(
            source_path,
            output_path,
            attachment_name=attachment_name,
            mime_type=mime_type,
        )
        written.append(Path(source_name))
    return written


def _sanitize_value(
    value: object,
    attachments: dict[str, tuple[str, str]],
) -> object:
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        sanitized: dict[str, object] = {}
        for raw_key, child in mapping.items():
            if not isinstance(raw_key, str):
                _fail()
            if raw_key == "statusDetails":
                sanitized[raw_key] = _sanitize_status_details(child)
            elif raw_key == "parameters":
                sanitized[raw_key] = _sanitize_parameters(child)
            elif raw_key == "attachments":
                sanitized[raw_key] = _sanitize_attachments(
                    child,
                    attachments,
                )
            else:
                sanitized[raw_key] = _sanitize_value(child, attachments)
        return sanitized
    if isinstance(value, list):
        sequence = cast("list[object]", value)
        return [_sanitize_value(item, attachments) for item in sequence]
    if isinstance(value, str):
        return _sanitize_report_text(value)
    if value is None or isinstance(value, int | float | bool):
        return value
    return _fail()


def _sanitize_status_details(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail()
    details = cast("dict[object, object]", value)
    message = details.get("message", "")
    trace = details.get("trace", "")
    exception_type = _exception_type(message, trace)
    sanitized: dict[str, object] = {
        "message": f"{exception_type}: {REDACTED_VALUE}",
        "trace": f"{exception_type}: traceback omitted by reporting policy",
    }
    for key in ("known", "muted", "flaky"):
        flag = details.get(key)
        if isinstance(flag, bool):
            sanitized[key] = flag
    return sanitized


def _sanitize_parameters(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        _fail()
    parameters: list[dict[str, object]] = []
    for raw_parameter in cast("list[object]", value):
        if not isinstance(raw_parameter, dict):
            _fail()
        parameter = cast("dict[object, object]", raw_parameter)
        name = parameter.get("name")
        if not isinstance(name, str) or not _SAFE_IDENTIFIER.fullmatch(name):
            _fail()
        safe_value = safe_reported_parameter_value(
            name,
            parameter.get("value", ""),
        )
        sanitized: dict[str, object] = {
            "name": name,
            "value": safe_value,
        }
        if safe_value == "<redacted>":
            sanitized["mode"] = "masked"
        else:
            for key in ("excluded", "mode"):
                candidate = parameter.get(key)
                if isinstance(candidate, bool | str):
                    sanitized[key] = candidate
        parameters.append(sanitized)
    return parameters


def _sanitize_attachments(
    value: object,
    attachments: dict[str, tuple[str, str]],
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        _fail()
    sanitized: list[dict[str, str]] = []
    for raw_attachment in cast("list[object]", value):
        if not isinstance(raw_attachment, dict):
            _fail()
        attachment = cast("dict[object, object]", raw_attachment)
        name = attachment.get("name")
        source = attachment.get("source")
        mime_type = attachment.get("type")
        if not all(
            isinstance(item, str) for item in (name, source, mime_type)
        ):
            _fail()
        assert isinstance(name, str)
        assert isinstance(source, str)
        assert isinstance(mime_type, str)
        if (
            not _SAFE_FILENAME.fullmatch(source)
            or _ATTACHMENT_MARKER not in source
            or not _SAFE_MIME_TYPE.fullmatch(mime_type)
        ):
            _fail()
        safe_name = _safe_attachment_name(name)
        descriptor = (safe_name, mime_type)
        if source in attachments and attachments[source] != descriptor:
            _fail()
        attachments[source] = descriptor
        sanitized.append(
            {
                "name": safe_name,
                "source": source,
                "type": mime_type,
            },
        )
    return sanitized


def _sanitize_attachment(
    source: Path,
    output: Path,
    *,
    attachment_name: str,
    mime_type: str,
) -> None:
    if not source.is_file() or source.is_symlink():
        _fail()
    if attachment_name == _DIAGNOSTIC_ATTACHMENT_NAME:
        _write_json(output, _sanitize_diagnostic(_load_object(source)))
        return
    if _is_text_attachment(source, mime_type):
        if "json" in mime_type or source.suffix.casefold() == ".json":
            _write_json(
                output,
                {
                    "sanitized": True,
                    "content": REDACTED_VALUE,
                },
            )
        else:
            output.write_text(_TEXT_ATTACHMENT_CONTENT, encoding="utf-8")
        return
    if _is_binary_attachment(mime_type):
        shutil.copyfile(source, output)
        return
    _fail()


def _sanitize_diagnostic(value: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    node_id = value.get("node_id")
    if isinstance(node_id, str):
        sanitized["node_id"] = _sanitize_report_text(node_id)
    phase = value.get("phase")
    if phase in {"setup", "call", "teardown"}:
        sanitized["phase"] = phase
    markers = value.get("markers")
    if isinstance(markers, list):
        safe_markers = [
            marker
            for marker in cast("list[object]", markers)
            if isinstance(marker, str) and _SAFE_IDENTIFIER.fullmatch(marker)
        ]
        sanitized["markers"] = sorted(safe_markers)
    duration = value.get("duration_seconds")
    if isinstance(duration, int | float) and duration >= 0:
        sanitized["duration_seconds"] = duration
    exception_type = value.get("exception_type")
    sanitized["exception_type"] = (
        exception_type
        if isinstance(exception_type, str)
        and _SAFE_EXCEPTION_TYPE.fullmatch(exception_type)
        else "TestFailure"
    )
    python_version = value.get("python_version")
    if isinstance(python_version, str) and re.fullmatch(
        r"[0-9]+(?:\.[0-9]+){1,3}",
        python_version,
    ):
        sanitized["python_version"] = python_version
    platform_value = value.get("platform")
    if isinstance(platform_value, str) and re.fullmatch(
        r"[A-Za-z0-9_. -]{1,100}",
        platform_value,
    ):
        sanitized["platform"] = platform_value
    run_id = value.get("github_run_id")
    if isinstance(run_id, str) and run_id.isdecimal():
        sanitized["github_run_id"] = run_id
    commit_sha = value.get("commit_sha")
    if isinstance(commit_sha, str) and _SAFE_SHA.fullmatch(commit_sha):
        sanitized["commit_sha"] = commit_sha
    if "node_id" not in sanitized or "phase" not in sanitized:
        _fail()
    return sanitized


def _load_object(path: Path) -> dict[str, object]:
    try:
        loaded = cast(
            object,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except (json.JSONDecodeError, OSError, UnicodeError) as error:
        message = "Invalid Allure JSON document."
        raise AllureResultSanitizationError(message) from error
    if not isinstance(loaded, dict):
        _fail()
    return cast("dict[str, object]", loaded)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _exception_type(*values: object) -> str:
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = value.split(":", maxsplit=1)[0].strip()
        if _SAFE_EXCEPTION_TYPE.fullmatch(candidate):
            return candidate
    return "TestFailure"


def _sanitize_report_text(value: str) -> str:
    sanitized = sanitize_full_text(value)
    sanitized = _PRIVATE_KEY.sub(REDACTED_VALUE, sanitized)
    sanitized = _SENSITIVE_ASSIGNMENT.sub(
        rf"\1{REDACTED_VALUE}",
        sanitized,
    )
    sanitized = _BEARER_TOKEN.sub(f"Bearer {REDACTED_VALUE}", sanitized)
    sanitized = _HOSTED_TOKEN.sub(REDACTED_VALUE, sanitized)
    return _LOCAL_PATH.sub("[LOCAL_PATH]", sanitized)


def _safe_attachment_name(value: str) -> str:
    if value in {
        "stdout",
        "stderr",
        "log",
        _DIAGNOSTIC_ATTACHMENT_NAME,
    }:
        return value
    return "Sanitized attachment"


def _is_text_attachment(path: Path, mime_type: str) -> bool:
    lowered = mime_type.casefold()
    return (
        lowered.startswith("text/")
        or any(term in lowered for term in ("json", "xml", "yaml"))
        or path.suffix.casefold() in _TEXT_EXTENSIONS
    )


def _is_binary_attachment(mime_type: str) -> bool:
    lowered = mime_type.casefold()
    return lowered.startswith(_BINARY_MIME_PREFIXES) or (
        lowered in _BINARY_MIME_TYPES
    )


def _remove_output(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path)
    except OSError as error:
        message = "Unable to remove an unsafe or stale report directory."
        raise AllureResultSanitizationError(message) from error


def _fail() -> NoReturn:
    message = "Raw Allure results failed validation."
    raise AllureResultSanitizationError(message)


if __name__ == "__main__":
    raise SystemExit(main())
