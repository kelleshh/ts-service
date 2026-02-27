from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.application.types import DatasetHandle
from tsfit.domain.value_objects import DatasetSchema, SeriesProfile


# Port
class SeriesProfiler(ABC):
    @abstractmethod
    def profile(self, ds: DatasetHandle, schema: DatasetSchema, *, min_train_ratio: float) -> SeriesProfile:
        raise NotImplementedError
