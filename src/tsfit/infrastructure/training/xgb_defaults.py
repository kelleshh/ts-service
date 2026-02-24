from __future__ import annotations
from typing import Any

def default_params() -> dict[str, Any]:
    '''
    Базовые гиперпараметры модели
    Выставляются дефолтные значения для всех параметров, которые пользователь не указал
    '''
    return {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        # в некоторых окружениях многопоточность может вести себя нестабильно поэтому дефолт делаем безопасным
        'n_jobs': 1,
    }


def merge_params(user_params: dict[str, Any]) -> dict[str, Any]:
    '''
    Склеивает дефолты и параметры пользователя
    '''

    merged = dict(default_params())
    merged.update(dict(user_params or {}))

    return merged













