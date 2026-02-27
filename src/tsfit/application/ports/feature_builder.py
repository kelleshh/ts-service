from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.application.types import DatasetHandle, SupervisedDataset
from tsfit.domain.value_objects import DatasetSchema, FeaturePlan


# Port
class FeatureBuilder(ABC):
    @abstractmethod
    def build_supervised(self, ds: DatasetHandle, schema: DatasetSchema, plan: FeaturePlan, *, horizon: int) -> DatasetHandle:
        '''
        Возвращает DatasetHandle, внутри которого лежит supervised table: features + target
        '''
        raise NotImplementedError

    @abstractmethod
    def split_xy(self, supervised: DatasetHandle, schema: DatasetSchema) -> SupervisedDataset:
        raise NotImplementedError

    @abstractmethod
    def build_last_x(self, ds: DatasetHandle, schema: DatasetSchema, plan: FeaturePlan) -> tuple[object, tuple[str, ...]]:
        '''
        Возвращает X для последней точки (без y) и feature_names
        '''
        raise NotImplementedError
