from __future__ import annotations

from typing import Any
import inspect
from xgboost import XGBRegressor

from tsfit.domain.metrics_invariants import is_higher_better
from tsfit.infrastructure.training.optuna.eval_helpers import pick_best


def _make_model_with_optional_early_stopping(
    *,
    params: dict[str, Any],
    primary_metric: str,
    early_stopping_rounds: int,
) -> XGBRegressor:
    '''
    Создает XGBRegressor и, если возможно, настраивает раннюю остановку

    
    !!! в разных версиях может отсутствовать early_stopping_round параметр, 
    тогда делаем через callbacks (либо через fit либо через конструктор)

    Мы делаем совместимость по сигнатурам инита
    '''

    init_sig = inspect.signature(XGBRegressor.__init__)
    maximize = bool(is_higher_better(primary_metric))

    # попытка 1: early_stopping_rounds в конструкторе
    if 'early_stopping_rounds' in init_sig.parameters:
        return XGBRegressor(
            **params,
            eval_metric=primary_metric,
            early_stopping_rounds=int(early_stopping_rounds),
        )

    # попытка 2: callbacks в конструкторе
    if 'callbacks' in init_sig.parameters:
        try:
            from xgboost.callback import EarlyStopping # а то ругается гад
        except Exception:
            EarlyStopping = None

        if EarlyStopping is not None:
            cb = EarlyStopping(
                rounds=int(early_stopping_rounds),
                save_best=True,
                maximize=maximize,
            )
            return XGBRegressor(
                **params,
                eval_metric=primary_metric,
                callbacks=[cb],
            )

    # если в конструкторе нет то создаем без ранней остановки
    # а раннюю остановку попробуем задать в fit() далее
    return XGBRegressor(
        **params,
        eval_metric=primary_metric,
    )


def _fit_with_optional_early_stopping(
    *,
    model: XGBRegressor,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    primary_metric: str,
    early_stopping_rounds: int,
) -> None:
    '''
    Запускает fit и, если версия поддерживает, добавляет раннюю остановку через fit()
    '''

    fit_sig = inspect.signature(model.fit)
    maximize = bool(is_higher_better(primary_metric))

    fit_kwargs: dict[str, Any] = {
        'eval_set': [(X_valid, y_valid)],
        'verbose': False,
    }

    # попытка 3: early_stopping_rounds в fit()
    if 'early_stopping_rounds' in fit_sig.parameters:
        fit_kwargs['early_stopping_rounds'] = int(early_stopping_rounds)
        model.fit(X_train, y_train, **fit_kwargs)
        return

    # попытка 4: callbacks в fit()
    if 'callbacks' in fit_sig.parameters:
        try:
            from xgboost.callback import EarlyStopping
        except Exception:
            EarlyStopping = None

        if EarlyStopping is not None:
            cb = EarlyStopping(
                rounds=int(early_stopping_rounds),
                save_best=True,
                maximize=maximize,
            )
            fit_kwargs['callbacks'] = [cb]
            model.fit(X_train, y_train, **fit_kwargs)
            return

    # иначе без ранней остановки ):
    model.fit(X_train, y_train, **fit_kwargs)



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
    - включаем раннюю остановку (если поддерживается)
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

        model = _make_model_with_optional_early_stopping(
            params=params,
            primary_metric=primary_metric,
            early_stopping_rounds=int(early_stopping_rounds),
        )

        _fit_with_optional_early_stopping(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            primary_metric=primary_metric,
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
        best_idx, best_val = pick_best(series, higher_is_better=higher) # выбираем лучшую итерацию и метрику там (которая главная метрика)

        candidates.append(
            {
                'learning_rate': float(lr),
                'best_iteration': int(best_idx + 1),
                'best_value': float(best_val),
            }
        )

    if not candidates:
        raise ValueError('learning_rate_grid пустой')

    # выбираем лучшего кандидата
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
