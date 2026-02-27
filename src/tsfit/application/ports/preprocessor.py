from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.application.types import DatasetHandle
from tsfit.domain.value_objects import DatasetSchema, PreprocessPlan


# Port
class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, ds: DatasetHandle, schema: DatasetSchema, plan: PreprocessPlan) -> DatasetHandle:
        raise NotImplementedError
