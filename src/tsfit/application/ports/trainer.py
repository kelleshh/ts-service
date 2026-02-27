from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.application.results import TrainingReport
from tsfit.application.types import SupervisedDataset
from tsfit.domain.value_objects import TrainingConfig


# Port
class ModelTrainer(ABC):
    @abstractmethod
    def train_with_walk_forward(
        self,
        *,
        full: SupervisedDataset,
        folds: list[tuple[SupervisedDataset, SupervisedDataset]],
        feature_names: tuple[str, ...],
        config: TrainingConfig,
    ) -> TrainingReport:
        raise NotImplementedError
