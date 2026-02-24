from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor

from tsfit.domain.metrics_invariants import is_higher_better

from tsfit.infrastructure.training.optuna.eval_helpers import pick_best


def pick_learning_rate_and_n_estimators(
    *,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    base_params: dict[str, Any],
    primary_metric: str,
    learning_rate_grid: list[float],
    max_estimators: int,
    early_stopping_rounds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    '''
    Этап 1: подбор связки learning_rate + n_estimators

    Алгоритм:
    - для каждого learning_rate обучаем модель с большим n_estimators
    - включаем раннюю остановку
    - на валидации ищем лучшую итерацию

    Возвращает:
    - dict fixed_params: {'learning_rate': ..., 'n_estimators': ...}
    - report: отчет по этапу 1
    '''

    higher = is_higher_better(primary_metric)

    candidates: list[dict[str, Any]] = []

    for lr in learning_rate_grid:
        params = dict(base_params)
        params['learning_rate'] = float(lr)
        params['n_estimators'] = int(max_estimators)
        params.setdefault('n_jobs', 1)

        model = XGBRegressor(**params, eval_metric=primary_metric)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=int(early_stopping_rounds),
        )

        evals = model.evals_result()
        valid_name = 'validation_0'
        if valid_name not in evals:
            raise ValueError('Не найден блок validation_0 в evals_result')

        valid_metrics = evals[valid_name]
        if primary_metric not in valid_metrics:
            known = sorted(valid_metrics.keys())
            raise ValueError(f'Модель не вернула метрику {primary_metric}. Доступно: {known}')

        series = valid_metrics[primary_metric]
        best_idx, best_val = pick_best(series, higher_is_better=higher)

        candidates.append(
            {
                'learning_rate': float(lr),
                'best_iteration': int(best_idx + 1),
                'best_value': float(best_val),
            }
        )

    if not candidates:
        raise ValueError('learning_rate_grid пустой')

    # ыыбираем лучшего кандидата
    chosen = candidates[0]
    for c in candidates[1:]:
        if higher:
            if c['best_value'] > chosen['best_value']:
                chosen = c
        else:
            if c['best_value'] < chosen['best_value']:
                chosen = c

    fixed = {
        'learning_rate': float(chosen['learning_rate']),
        'n_estimators': int(chosen['best_iteration']),
    }

    report = {
        'learning_rates': list(learning_rate_grid),
        'candidates': candidates,
        'chosen_learning_rate': float(chosen['learning_rate']),
        'chosen_n_estimators': int(chosen['best_iteration']),
        'chosen_best_value': float(chosen['best_value']),
    }

    return fixed, report
