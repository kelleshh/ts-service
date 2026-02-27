from __future__ import annotations

from typing import Any
from dataclasses import dataclass

# Value Object
@dataclass(frozen=True, slots=True)
class FitCommand:
    idempotency_key: str
    rows: list[dict[str, Any]]
    timestamp_col: str
    target_col: str
    exog_cols: tuple[str, ...]
    horizon: int

    primary_metric: str
    metrics: tuple[str, ...]
    model_params: dict[str, Any]
    early_stopping_rounds: int
    n_estimators_cap: int


# Value Object
class FitAutoCommand(FitCommand):
    
    n_trials: int
    timeout_sec: int | None


# Value Object
@dataclass(frozen=True, slots=True)
class PredictCommand:
    model_id: str
    rows: list[dict]
