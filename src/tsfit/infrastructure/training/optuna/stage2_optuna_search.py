from __future__ import annotations

from typing import Any

import optuna
from xgboost import XGBRegressor

from tsfit.domain.metrics_invariants import is_higher_better
from tsfit.domain.tuning_policy import DatasetProfile, TuningPolicy

from tsfit.infrastructure.training.optuna.search_space import suggest_params


def run_optuna_search(
    *,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    base_params: dict[str, Any],
    fixed_params: dict[str, Any],
    primary_metric: str,
    profile: DatasetProfile,
    policy: TuningPolicy,
    n_trials_requested: int,
    timeout_sec: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    '''
    Этап 2: optuna подбирает остальные параметры

    learning_rate и n_estimators считаются уже выбранными (fixed_params)

    Возвращает:
    - best_params: подобранные параметры
    - report: отчет по этапу 2
    '''

    direction = 'maximize' if is_higher_better(primary_metric) else 'minimize'
    n_trials_used = int(min(int(n_trials_requested), int(policy.n_trials_cap)))

    def objective(trial: optuna.Trial) -> float:
        params = dict(base_params)
        params.update(dict(fixed_params))
        params.update(suggest_params(trial, policy=policy, profile=profile))

        params.setdefault('n_jobs', 1)

        model = XGBRegressor(
            **params,
            eval_metric=primary_metric,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        evals_result = model.evals_result()
        valid_name = 'validation_0'
        if valid_name not in evals_result:
            raise ValueError('Не найден блок validation_0 в evals_result')

        valid_metrics = evals_result[valid_name]
        if primary_metric not in valid_metrics:
            known = sorted(valid_metrics.keys())
            raise ValueError(f'Модель не вернула метрику {primary_metric}. Доступно: {known}')

        series = valid_metrics[primary_metric]
        if not series:
            raise ValueError(f'Пустая история значений метрики {primary_metric}')

        return float(series[-1])

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials_used, timeout=timeout_sec)

    best_params = dict(study.best_trial.params)

    report = {
        'direction': direction,
        'n_trials_requested': int(n_trials_requested),
        'n_trials_used': int(n_trials_used),
        'n_trials_ran': int(len(study.trials)),
        'best_value': float(study.best_value),
        'best_params': best_params,
    }

    return best_params, report
