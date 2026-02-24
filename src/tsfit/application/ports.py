from __future__ import annotations
from dataclasses import dataclass

from typing import Protocol, Any

from tsfit.domain.entities import TrainingRunEntity
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
    '''
    Порт для хранения и получения TrainingRunEntity.
    '''
    def save(self, run: TrainingRunEntity) -> None: ...
    def get(self, run_id: str) -> TrainingRunEntity | None: ...
    def find_by_idempotency(self, key: str) -> TrainingRunEntity | None: ...


class DatasetParser(Protocol):
    '''
    Порт для преобразования входного JSON в табличный формат.
    '''
    def parse(self, rows: list[dict[str, Any]], schema: DatasetSchemaValueObject) -> Frame:
        '''парсит JSON ы табличный формат и делает строгие проверки временнОй оси'''
        ...


class TimeSeriesDatasetBuilder(Protocol):
    '''
    Порт для построения признаков и разреза по времени.
    '''
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
    '''
    Порт обучения модели
    '''
    def train_and_eval(
        self,
        dataset: BuiltDataset,
        training: Any,
    ) -> dict[str, Any]:
        '''Обучает модель и возвращает результат (метрики и тд)'''
        ...


class HyperparameterTuner(Protocol):
    '''
    Порт для подбора гиперпараметров (например через optuna)

    application слой знает только про сам факт подбора и контракт вход/выход
    конкретная библиотека (optuna) живет в инфраструктуре
    '''

    def tune(
        self,
        dataset: BuiltDataset,
        base_params: dict[str, Any],
        primary_metric: str,
        tuning: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        '''
        Возвращает:
        - best_params: словарь параметров, которые нужно ДОБАВИТЬ/ПЕРЕОПРЕДЕЛИТЬ в base_params
        - report: короткий отчет о подборе (можно сохранять в run.result)
        '''
        ...