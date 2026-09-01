"""Judge correction request domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JudgeCorrectionRequest:
    """Data needed for one judge-output correction attempt."""

    invalid_response: str
    validation_errors: tuple[str, ...]
