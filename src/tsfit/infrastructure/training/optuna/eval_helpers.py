from __future__ import annotations

from typing import Iterable


def pick_best(value_series: Iterable[float], higher_is_better: bool) -> tuple[int, float]:
    '''
    Возвращает лучшие индекс и значение (best_index, best_value) по истории метрики
    Параметры:

    best_index: индекс в списке (0..)
    higher_is_better:
    - True: выбираем максимум
    - False: выбираем минимум
    '''
    values = list(value_series)
    if not values:
        raise ValueError('Пустая история значений метрики')
    
    # поиск best val и best idx
    best_idx = 0
    best_val = float(values[0])
    for i, v in enumerate(values[1:], start=1):
        fv = float(v)
        if higher_is_better:
            if fv > best_val:
                best_val = fv
                best_idx = i
        else:
            if fv < best_val:
                best_val = fv
                best_idx = i
            
    return best_idx, best_val