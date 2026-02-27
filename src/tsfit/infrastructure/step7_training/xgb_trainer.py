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
    p.setdefault('learning_rate', 0.05)
    p.setdefault('subsample', 0.8)
    p.setdefault('colsample_bytree', 0.8)
    p.setdefault('min_child_weight', 1.0)
    p.setdefault('reg_alpha', 0.0)
    p.setdefault('reg_lambda', 1.0)
    p.setdefault('gamma', 0.0)
    p.setdefault('seed', 42)

    lr = float(p.pop('learning_rate')) if 'learning_rate' in p else float(p.get('eta', 0.05))  # type: ignore
    p['eta'] = lr

    # xgboost train не любит лишние ключи
    if 'n_estimators' in p:
        p.pop('n_estimators')

    return p


def _make_feval(metric: str, *, eps: float):
    def feval(preds: np.ndarray, dmat: xgb.DMatrix):
        y = dmat.get_label()
        out = compute_all(metrics=[metric], y_true=y, y_pred=preds, eps=eps)
        return metric, float(out[metric])

    return feval


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

        fold_train_metrics: list[dict[str, float]] = []
        fold_valid_metrics: list[dict[str, float]] = []
        best_iters: list[int] = []

        for train_sd, valid_sd in folds:
            dtrain = xgb.DMatrix(train_sd.x, label=train_sd.y, feature_names=list(feature_names))
            dvalid = xgb.DMatrix(valid_sd.x, label=valid_sd.y, feature_names=list(feature_names))

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=int(config.n_estimators_cap),
                evals=[(dtrain, 'train'), (dvalid, 'valid')],
                early_stopping_rounds=int(config.early_stopping_rounds),
                verbose_eval=False,
                custom_metric=_make_feval(config.primary_metric, eps=config.mape_eps),
                maximize=False,
            )

            best_it = int(getattr(booster, 'best_iteration', config.n_estimators_cap - 1))
            best_iters.append(best_it + 1)

            yhat_train = booster.predict(dtrain, iteration_range=(0, best_it + 1))
            yhat_valid = booster.predict(dvalid, iteration_range=(0, best_it + 1))

            fold_train_metrics.append(
                compute_all(metrics=list(config.metrics), y_true=train_sd.y, y_pred=yhat_train, eps=config.mape_eps) # type: ignore
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
        dfull = xgb.DMatrix(full.x, label=full.y, feature_names=list(feature_names))
        final = xgb.train(
            params=params,
            dtrain=dfull,
            num_boost_round=best_n,
            evals=[(dfull, 'train')],
            verbose_eval=False,
        )

        gain = final.get_score(importance_type='gain')
        fi: dict[str, float] = {name: float(gain.get(name, 0.0)) for name in feature_names}

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
