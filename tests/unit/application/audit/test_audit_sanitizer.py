"""Tests for audit sanitization helpers."""

import json

import pytest

from agent_team.application.audit.audit_sanitizer import (
    MAX_AUDIT_EXCERPT_LENGTH,
    omit_hidden_reasoning,
    sanitize_text,
    sanitize_tool_arguments,
)


class TestAuditSanitizer:
    """Audit sanitizer behavior tests."""

    def test_text_excerpts_are_truncated(self) -> None:
        """Trim long excerpts to the documented maximum length."""
        excerpt = sanitize_text("x" * (MAX_AUDIT_EXCERPT_LENGTH + 40))

        assert len(excerpt) == MAX_AUDIT_EXCERPT_LENGTH
        assert excerpt.endswith("...")

    def test_secret_like_keys_are_redacted(self) -> None:
        """Redact values for sensitive key names."""
        _hash, preview = sanitize_tool_arguments(
            "create_feature",
            {
                "title": "Secure auth",
                "authorization": "Bearer secret-token",
                "nested": {"api_key": "private-key"},
            },
        )

        assert "Bearer secret-token" not in preview
        assert "private-key" not in preview
        assert "[REDACTED]" in preview

    def test_hidden_reasoning_is_omitted_from_excerpts(self) -> None:
        """Remove local-model thinking blocks before audit storage."""
        excerpt = sanitize_text("<think>private reasoning</think>Visible.")

        assert "private reasoning" not in excerpt
        assert excerpt == "[hidden reasoning omitted]Visible."

    def test_hidden_reasoning_is_omitted_from_visible_output(self) -> None:
        """Remove local-model thinking blocks before user output."""
        output = omit_hidden_reasoning("<think>hidden</think>Visible.")

        assert output == "[hidden reasoning omitted]Visible."

    def test_create_task_arguments_keep_valid_json_without_description(
        self,
    ) -> None:
        """Store effective task metadata without raw task descriptions."""
        description = "Implement sensitive backend details. " * 40

        _hash, preview = sanitize_tool_arguments(
            "create_task",
            {
                "feature_id": 1,
                "title": "Implement order search endpoint",
                "description": description,
                "assigned_role": "backend_developer",
                "status": "pending",
            },
        )

        parsed = json.loads(preview)
        assert parsed["feature_id"] == 1
        assert parsed["assigned_role"] == "backend_developer"
        assert parsed["status"] == "pending"
        assert parsed["description_length"] == len(description)
        assert "description_hash" in parsed
        assert "Implement sensitive backend details" not in preview

    def test_symbol_names_are_hashed_in_audit_previews(self) -> None:
        """Avoid retaining repository symbol names in audit previews."""
        _hash, preview = sanitize_tool_arguments(
            "find_symbol",
            {"name": "PrivateBillingService.rotate_credential"},
        )

        parsed = json.loads(preview)
        assert "PrivateBillingService" not in preview
        assert parsed["name_length"] == len(
            "PrivateBillingService.rotate_credential",
        )
        assert len(parsed["name_hash"]) == 64

    @pytest.mark.parametrize(
        "role",
        [
            "backend_developer",
            "frontend_developer",
            "qa_engineer",
            "code_reviewer",
        ],
    )
    def test_create_task_sanitizes_sequential_role_arguments(
        self,
        role: str,
    ) -> None:
        """Keep effective task arguments observable for every target role."""
        description = f"Long {role} task description. " * 40

        _hash, preview = sanitize_tool_arguments(
            "create_task",
            {
                "feature_id": 1,
                "title": f"{role} task",
                "description": description,
                "assigned_role": role,
                "status": "pending",
            },
        )

        parsed = json.loads(preview)
        assert parsed["feature_id"] == 1
        assert parsed["assigned_role"] == role
        assert parsed["status"] == "pending"
        assert parsed["title"] == f"{role} task"
        assert parsed["description_hash"]
        assert parsed["description_length"] == len(description)
        assert "description" not in parsed
        assert description not in preview
