from __future__ import annotations

import statistics

import numpy as np
import xgboost as xgb

from tsfit.application.ports import ModelTrainer
from tsfit.application.results import MetricsAgg, MetricsReport, TrainingReport
from tsfit.application.types import SupervisedDataset
from tsfit.domain.value_objects import TrainingConfig
from tsfit.infrastructure.step8_evaluation.metrics import compute_all


def _normalize_params(model_params: dict[str, object]) -> dict[str, object]:
    p = dict(model_params)

    # базовые дефолты
    p.setdefault('objective', 'reg:squarederror')
    p.setdefault('tree_method', 'hist')
    p.setdefault('max_depth', 6)
    p.setdefault('subsample', 0.8)
    p.setdefault('colsample_bytree', 0.8)
    p.setdefault('min_child_weight', 1.0)
    p.setdefault('reg_alpha', 0.0)
    p.setdefault('reg_lambda', 1.0)
    p.setdefault('gamma', 0.0)
    p.setdefault('seed', 42)

    # learning_rate и eta это синонимы поэтому ставим просто eta
    if 'eta' in p:
        lr = float(p['eta'])  # type: ignore
    elif 'learning_rate' in p:
        lr = float(p['learning_rate'])  # type: ignore
    else:
        lr = 0.05
    p['eta'] = lr
    p.pop('learning_rate', None)

    # xgboost train не любит лишние ключи
    p.pop('n_estimators', None)

    return p


def _make_feval(metric: str, *, eps: float):
    def feval(preds: np.ndarray, dmat: xgb.DMatrix):
        y = dmat.get_label()
        out = compute_all(metrics=[metric], y_true=y, y_pred=preds, eps=eps)
        return metric, float(out[metric])

    return feval


def _time_weights(n: int) -> np.ndarray:
    # мягкое усиление веса последних точек (полезно для дата дрифта)
    if n <= 1:
        return np.ones((n,), dtype='float64')
    return np.linspace(0.3, 1.0, n, dtype='float64')


_BUILTIN_PRIMARY_METRICS = {'rmse', 'mae'}


def _prepare_eval(primary_metric: str, *, mape_eps: float) -> tuple[dict[str, object], object | None, bool]:
    params_update: dict[str, object] = {}
    custom_metric: object | None = None

    if primary_metric in _BUILTIN_PRIMARY_METRICS:
        params_update['eval_metric'] = primary_metric
    else:
        params_update['disable_default_eval_metric'] = 1
        custom_metric = _make_feval(primary_metric, eps=mape_eps)

    return params_update, custom_metric, False


# Adapter (Infrastructure)
class XGBoostTrainer(ModelTrainer):
    def train_with_walk_forward(
        self,
        *,
        full: SupervisedDataset,
        folds: list[tuple[SupervisedDataset, SupervisedDataset]],
        feature_names: tuple[str, ...],
        config: TrainingConfig,
    ) -> TrainingReport:
        params = _normalize_params(config.model_params)
        eval_update, custom_metric, maximize = _prepare_eval(config.primary_metric, mape_eps=config.mape_eps)
        params.update(eval_update)

        fold_train_metrics: list[dict[str, float]] = []
        fold_valid_metrics: list[dict[str, float]] = []
        best_iters: list[int] = []

        for train_sd, valid_sd in folds:
            w_train = _time_weights(int(train_sd.n_rows))

            dtrain = xgb.DMatrix(train_sd.x, label=train_sd.y, feature_names=list(feature_names), weight=w_train)
            dvalid = xgb.DMatrix(valid_sd.x, label=valid_sd.y, feature_names=list(feature_names))

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=int(config.n_estimators_cap),
                evals=[(dtrain, 'train'), (dvalid, 'valid')],
                early_stopping_rounds=int(config.early_stopping_rounds),
                verbose_eval=False,
                custom_metric=custom_metric,
                maximize=maximize,
            )

            best_it = int(getattr(booster, 'best_iteration', config.n_estimators_cap - 1))
            best_iters.append(best_it + 1)

            yhat_train = booster.predict(dtrain, iteration_range=(0, best_it + 1))
            yhat_valid = booster.predict(dvalid, iteration_range=(0, best_it + 1))
            # считаем трейн метрики сопоставимыми с vaild
            tail = int(valid_sd.n_rows)
            y_true_train = train_sd.y[-tail:] # type: ignore
            y_pred_train = yhat_train[-tail:]

            fold_train_metrics.append(
                compute_all(metrics=list(config.metrics), y_true=y_true_train, y_pred=y_pred_train, eps=config.mape_eps)
            )
            fold_valid_metrics.append(
                compute_all(metrics=list(config.metrics), y_true=valid_sd.y, y_pred=yhat_valid, eps=config.mape_eps) # type: ignore
            )

        # агрегируем метрики
        train_agg: dict[str, MetricsAgg] = {}
        valid_agg: dict[str, MetricsAgg] = {}
        for m in config.metrics:
            tr = np.array([d[m] for d in fold_train_metrics], dtype='float64')
            va = np.array([d[m] for d in fold_valid_metrics], dtype='float64')
            train_agg[m] = MetricsAgg(mean=float(tr.mean()), std=float(tr.std(ddof=0)))
            valid_agg[m] = MetricsAgg(mean=float(va.mean()), std=float(va.std(ddof=0)))

        best_n = int(statistics.median(best_iters))
        best_n = max(1, min(best_n, int(config.n_estimators_cap)))

        # финальная модель на всём датасете
        w_full = _time_weights(int(full.n_rows))
        dfull = xgb.DMatrix(full.x, label=full.y, feature_names=list(feature_names), weight=w_full)
        final = xgb.train(
            params=params,
            dtrain=dfull,
            num_boost_round=best_n,
            evals=[(dfull, 'train')],
            verbose_eval=False,
        )

        gain = final.get_score(importance_type='gain')
        fi: dict[str, float] = {name: float(gain.get(name, 0.0)) for name in feature_names} # type: ignore

        report = TrainingReport(
            model=final,
            model_params=dict(params),
            training_params={
                'early_stopping_rounds': int(config.early_stopping_rounds),
                'n_estimators_cap': int(config.n_estimators_cap),
                'n_estimators_final': int(best_n),
                'primary_metric': config.primary_metric,
                'metrics': list(config.metrics),
            },
            metrics=MetricsReport(train=train_agg, valid=valid_agg),
            feature_names=tuple(feature_names),
            feature_importance_gain=fi,
        )
        return report