from __future__ import annotations

from dataclasses import dataclass


# Value Object
@dataclass(frozen=True, slots=True)
class SeriesProfile:
    '''
    Описание временного ряда, которое потом используется доменной политикой
    для выбора FeaturePlan, CVPLan и тд
    '''
    n_train: int # размер обучающей выборки
    is_regular: bool # регулярный ли ряд по времени (>95% одинаковых интервалов)
    has_trend: bool # имеет ли ряд тренд (если да то есть смысл в time_index и глубже лаги)
    season_period: int | None # найденный период сезонности
    is_exponential: bool # имеет ли ряд экспоненциальную форму распределения
