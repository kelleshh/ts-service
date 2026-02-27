from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.application.types import DatasetHandle
from tsfit.application.results import TrainingReport
from tsfit.domain.value_objects import CVPlan, DatasetSchema, FeaturePlan, TrainingConfig


# Port
class ModelTuner(ABC):
    @abstractmethod
    def tune_and_train(
        self,
        supervised: DatasetHandle,
        schema: DatasetSchema,
        feature_plan: FeaturePlan,
        cv_plan: CVPlan,
        config: TrainingConfig,
        *,
        horizon: int,
        n_trials: int,
        timeout_sec: int | None,
    ) -> TrainingReport:
        raise NotImplementedError
