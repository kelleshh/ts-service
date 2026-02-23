from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBRegressor

from tsfit.application.ports import BuiltDataset, ModelTrainer
from tsfit.domain.spec import TrainingConfig
from tsfit.infrastructure.training.tuning_optuna import tune_xgb_params

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


class XGBModelTrainer(ModelTrainer):
    '''
    Реализация обучателя для XGBoost (с поддержкой optuna)

    Возвращает словарь с мерриками, важностью фичей и параметрами
    '''

    def train_and_eval(self, dataset: BuiltDataset, training: TrainingConfig) -> dict[str, Any]:

        X_train = dataset.X_train
        y_train = dataset.y_train
        X_valid = dataset.X_valid
        y_valid = dataset.y_valid
        feature_names = dataset.feature_names

        user_params = dict(training.xgb_params or {})

        # в случае если пользователь выключил тюнинг оптуной юзаем это
        defaults: dict[str, Any] = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }

        merged = {**defaults, **user_params}

        tuning_report: dict[str, Any] | None = None
        if training.tuning.enabled:
            best_params, tuning_report = tune_xgb_params(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                base_params=merged,
                primary_metric=training.primary_metric,
                n_trials=training.tuning.n_trials,
                timeout_sec=training.tuning.timeout_sec,
                early_stopping_rounds=training.tuning.early_stopping_rounds,
            )
            merged = {**merged, **best_params}

        model = XGBRegressor(**merged)
        if training.tuning.enabled:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
                early_stopping_rounds=training.tuning.early_stopping_rounds,
            )
        else:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        pred = model.predict(X_valid)
        yv = np.asarray(y_valid)
        pv = np.asarray(pred)

        metrics: dict[str, float] = {}
        for m in training.metrics:
            if m == 'rmse':
                metrics[m] = _rmse(yv, pv)
            elif m == 'mae':
                metrics[m] = _mae(yv, pv)
            else:
                raise ValueError(f'Неподдерживаемая метрика: {m}')

        # importance (топ-50 фичей)
        feature_importance: dict[str, float] = {}
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            feature_importance = {
                str(name): float(val) for name, val in zip(feature_names, fi)
            }
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:50]
            )

        out: dict[str, Any] = {
            'metrics': metrics,
            'primary_metric': training.primary_metric,
            'primary_metric_value': float(metrics[training.primary_metric]),
            'feature_importance': feature_importance,
            'model_params': merged,
        }
        if tuning_report is not None:
            out['tuning'] = tuning_report
        return out