"""Task handoff field limit model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskHandoffLimits:
    """Centralized bounds for persisted task handoff data."""

    implementation_summary_chars: int = 1200
    changed_path_count: int = 24
    reused_symbol_count: int = 24
    new_symbol_count: int = 24
    reuse_notes_chars: int = 800
    checks_attempted_count: int = 12
    limitations_chars: int = 600
    next_action_chars: int = 400
    item_chars: int = 240
    total_chars: int = 5000


DEFAULT_TASK_HANDOFF_LIMITS = TaskHandoffLimits()
