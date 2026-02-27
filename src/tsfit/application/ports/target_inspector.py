from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.application.types import DatasetHandle
from tsfit.domain.value_objects import DatasetSchema


# Port
class TargetInspector(ABC):
    @abstractmethod
    def frac_abs_y_lt_eps(self, ds: DatasetHandle, schema: DatasetSchema, *, eps: float) -> float:
        raise NotImplementedError
