from __future__ import annotations

from dataclasses import dataclass


# Value Object
@dataclass(frozen=True, slots=True)
class FitResult:
    model_id: str
    created: bool
    summary: dict


# Value Object
@dataclass(frozen=True, slots=True)
class PredictResult:
    prediction: float
