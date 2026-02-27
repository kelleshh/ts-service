from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricsAgg:
    '''
    Агрегированное значение метрики по кросс валидации

    mean - среднее по фолдам
    std - стандартное отклонение по фолдам
    '''
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class MetricsReport:
    '''
    Отчет по метрикам для train и valid частей кросс валидации
    ключ словаря - имя метрики, значение - агрегированные статистики по фолдам
    '''
    train: dict[str, MetricsAgg]
    valid: dict[str, MetricsAgg]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    '''
    Полный результат обучения модели
    '''
    model: object # обученная модель
    model_params: dict[str, object] # параметры модели (возможно после тюнинга)
    training_params: dict[str, object] # параметры процесса обучения
    metrics: MetricsReport # агрегированные метрики
    feature_names: tuple[str, ...] # кортеж фичей
    feature_importance_gain: dict[str, float] # важности признаков (по gain)
