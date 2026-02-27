from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetHandle:
    '''Непрозрачная обертка над датасетом (внутри infra может быть pandas.DataFrame)'''

    payload: object # какой то массив (любой)


@dataclass(frozen=True, slots=True)
class Fold: # фолды
    train: DatasetHandle # трейн выборка
    valid: DatasetHandle # валидационная выборка


@dataclass(frozen=True, slots=True)
class SupervisedDataset:
    '''Непрозрачный набор для обучения (внутри infra могут быть матрицы/df)'''

    x: object # фичи
    y: object #таргет
    feature_names: tuple[str, ...] # имена фичей
    n_rows: int # колво строк
    n_features: int # колво фичей
