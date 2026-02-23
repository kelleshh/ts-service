from __future__ import annotations
from dataclasses import dataclass

from typing import Protocol, Any

from tsfit.domain.entites import TrainingRunEntity
from tsfit.domain.value_objects import DatasetSchemaValueObject, TimeSeriesConfigValueObject

Frame = Any # application не должен знать про pandas/polars и pl.DataFrame/pd.DataFrame
Series = Any # аналогичн для series

@dataclass(frozen=True)
class BuiltDataset:
    '''Результат построения supervised-датасета для обучения.

    Типы намеренно абстрактные Frame/Series, чтобы application слой не зависел от pandas/polars/numpy.
    '''

    X_train: Frame
    y_train: Series
    X_valid: Frame
    y_valid: Series
    feature_names: list[str]


class TrainingRunRepository(Protocol):
    def save(self, run: TrainingRunEntity) -> None: ...
    def get(self, run_id: str) -> TrainingRunEntity | None: ...
    def find_by_idempotency(self, key: str) -> TrainingRunEntity | None: ...


class DatasetParser(Protocol):
    def parse(self, rows: list[dict[str, Any]], schema: DatasetSchemaValueObject) -> Frame:
        '''парсит JSON ы табличный формат и делает строгие проверки временнОй оси'''
        ...


class TimeSeriesDatasetBuilder(Protocol):
    def build_train_valid(
        self,
        frame: Frame,
        schema: DatasetSchemaValueObject,
        cfg: TimeSeriesConfigValueObject,
    ) -> BuiltDataset:
        '''
        Строит supervised-датасет:
        - признаки (лаги и др.)
        - цель на horizon: y(t+h)
        - разрез train/valid по времени
        Возвращает BuiltDataset
        '''
        ...


class ModelTrainer(Protocol):
    def train_and_eval(
        self,
        dataset: BuiltDataset,
        training: Any,
    ) -> dict[str, Any]:
        '''Обучает модель и возвращает результат (метрики и тд)'''
        ...