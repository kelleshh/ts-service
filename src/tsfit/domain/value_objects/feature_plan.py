from __future__ import annotations

from dataclasses import dataclass


# Value Object
@dataclass(frozen=True, slots=True)
class FeaturePlan:
    '''
    План-заготовка для фиче-инжиниринга
    Определяется в рамках политики обработки для конкретного датасета
    '''
    max_lag: int # сколько лагов таргета строить максимум
    rolling_windows: tuple[int, ...] # окна для скользящего среднего
    rolling_std_windows: tuple[int, ...] # окна для скользящего стандартного отклонения
    add_delta_t: bool # добавить ли delta_t (если ряд нерегулируемый)
    add_time_index: bool # добавить ли t_idx (монотонный индекс времени)
    season_period: int | None # период сезонности
    add_season_sin_cos: bool # добавить ли синус косинус сезонности
    add_exp_features: bool # добавить log/exp признаки (если ряд похож на экспоненту)
