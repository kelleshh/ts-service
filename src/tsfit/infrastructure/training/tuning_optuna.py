from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from xgboost import XGBRegressor

from tsfit.application.ports import BuiltDataset, HyperparameterTuner
from tsfit.domain.value_objects import TuningConfigValueObject
from tsfit.application.ports import BuiltDataset, HyperparameterTuner
from tsfit.domain.value_objects import TuningConfigValueObject

from tsfit.domain.metrics_invariants import is_higher_better


def tune_xgb_params(
    *,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    base_params: dict[str, Any],
    primary_metric: str,
    n_trials: int,
    timeout_sec: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    '''
    Подбор гиперпараметров на holdout (X_valid).

    Тюнинг с помощью инструмента optuna

    Возвращает (best_params, tuning_report).
    '''


    # определение направления оптимизации метрики
    direction = 'maximize' if is_higher_better(primary_metric) else 'minimize'


    def objective(trial: optuna.Trial) -> float:
        params = dict(base_params)
        params.pop('eval_metric', None)

        params.update(
            {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            }
        )

        model = XGBRegressor(**params,
                             eval_metric = primary_metric)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        evals_result = model.evals_result()
        valid_name = 'validation_0'
        if valid_name not in evals_result:
            raise ValueError('XGBoost не вернул evals_result для validation_0')

        valid_metrics = evals_result[valid_name]
        if primary_metric not in valid_metrics:
            known = sorted(valid_metrics.keys())
            raise ValueError(
                f'XGBoost не вернул метрику {primary_metric}. Доступно: {known}'
            )

        series = valid_metrics[primary_metric]
        if not series:
            raise ValueError(f'Пустая история значений метрики {primary_metric}')

        return float(series[-1])

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    best_params = dict(study.best_trial.params)
    report = {
        'n_trials': len(study.trials),
        'best_value': float(study.best_value),
        'best_params': best_params,
    }
    return best_params, report



class OptunaXGBTuner(HyperparameterTuner):
    '''
    Инфраструктурный сервис подбора гиперпараметра через optuna
    '''

    def tune(
        self,
        dataset: BuiltDataset,
        base_params: dict[str, Any],
        primary_metric: str,
        tuning: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        if isinstance(tuning, TuningConfigValueObject):
            cfg = tuning
        else:
            raise TypeError('tuning должен быть TuningConfigValueObject')

        return tune_xgb_params(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_valid=dataset.X_valid,
            y_valid=dataset.y_valid,
            base_params=base_params,
            primary_metric=primary_metric,
            n_trials=cfg.n_trials,
            timeout_sec=cfg.timeout_sec,
        )
