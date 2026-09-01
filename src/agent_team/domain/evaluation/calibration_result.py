"""Judge calibration result domain model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Agreement metrics between human labels and judge grades."""

    verdict_agreement: float
    dimension_exact_agreement: dict[str, float]
    dimension_mean_absolute_error: dict[str, float]
    judge_ambiguity_rate: float
    disagreements: tuple[str, ...]
