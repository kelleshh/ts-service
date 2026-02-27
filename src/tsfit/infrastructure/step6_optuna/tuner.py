from __future__ import annotations

import statistics

import numpy as np
import xgboost as xgb

from tsfit.domain.value_objects.meta import ModelMeta
from tsfit.application.ports import ModelTuner
from tsfit.application.results import MetricsAgg, MetricsReport, TrainingReport
from tsfit.domain.exceptions import ValidationError
from tsfit.domain.value_objects import CVPlan, DatasetSchema, FeaturePlan, TrainingConfig
from tsfit.infrastructure.step5_cross_validation.walk_forward import DefaultWalkForwardSplitter
from tsfit.infrastructure.step8_evaluation.metrics import compute_all
from tsfit.infrastructure.step4_feature_generation.feature_builder import PandasFeatureBuilder
from tsfit.infrastructure.step7_training.xgb_trainer import _normalize_params


# Adapter (Infrastructure)
class OptunaTuner(ModelTuner):
    def __init__(self) -> None:
        self._splitter = DefaultWalkForwardSplitter()
        self._fb = PandasFeatureBuilder()

    def tune_and_train(
        self,
        supervised,
        schema: DatasetSchema,
        feature_plan: FeaturePlan, # он, кажется тут не нужен
        cv_plan: CVPlan,
        config: TrainingConfig,
        *,
        horizon: int,# он, кажется тут не нужен
        n_trials: int,
        timeout_sec: int | None,
    ) -> TrainingReport:
        try:
            import optuna
        except Exception as e:  # pragma: no cover
            raise RuntimeError('optuna не установлен') from e

        full_sd = self._fb.split_xy(supervised, schema)
        folds_h = self._splitter.split(supervised, cv_plan)
        folds = [(self._fb.split_xy(f.train, schema), self._fb.split_xy(f.valid, schema)) for f in folds_h]

        feature_names = full_sd.feature_names
        base = _normalize_params(config.model_params)

        def objective(trial: 'optuna.Trial') -> float:
            params = dict(base)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 12)
            params['min_child_weight'] = trial.suggest_float('min_child_weight', 0.1, 10.0, log=True)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            params['eta'] = trial.suggest_float('eta', 0.01, 0.2, log=True)

            vals: list[float] = []
            for train_sd, valid_sd in folds:
                dtrain = xgb.DMatrix(train_sd.x, label=train_sd.y, feature_names=list(feature_names))
                dvalid = xgb.DMatrix(valid_sd.x, label=valid_sd.y, feature_names=list(feature_names))

                booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=int(config.n_estimators_cap),
                    evals=[(dvalid, 'valid')],
                    early_stopping_rounds=int(config.early_stopping_rounds),
                    verbose_eval=False,
                )

                best_it = int(getattr(booster, 'best_iteration', config.n_estimators_cap - 1))
                yhat = booster.predict(dvalid, iteration_range=(0, best_it + 1))
                m = compute_all(
                    metrics=[config.primary_metric],
                    y_true=valid_sd.y, # type: ignore
                    y_pred=yhat,
                    eps=config.mape_eps,
                )[config.primary_metric]
                vals.append(float(m))

            return float(np.mean(vals))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=int(n_trials), timeout=timeout_sec)

        best_params = dict(base)
        best_params.update(study.best_params)

        # теперь финально обучаем как обычный trainer
        fold_train_metrics: list[dict[str, float]] = []
        fold_valid_metrics: list[dict[str, float]] = []
        best_iters: list[int] = []

        for train_sd, valid_sd in folds:
            dtrain = xgb.DMatrix(train_sd.x, label=train_sd.y, feature_names=list(feature_names))
            dvalid = xgb.DMatrix(valid_sd.x, label=valid_sd.y, feature_names=list(feature_names))

            booster = xgb.train(
                params=best_params,
                dtrain=dtrain,
                num_boost_round=int(config.n_estimators_cap),
                evals=[(dtrain, 'train'), (dvalid, 'valid')],
                early_stopping_rounds=int(config.early_stopping_rounds),
                verbose_eval=False,
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

        train_agg: dict[str, MetricsAgg] = {}
        valid_agg: dict[str, MetricsAgg] = {}
        for m in config.metrics:
            tr = np.array([d[m] for d in fold_train_metrics], dtype='float64')
            va = np.array([d[m] for d in fold_valid_metrics], dtype='float64')
            train_agg[m] = MetricsAgg(mean=float(tr.mean()), std=float(tr.std(ddof=0)))
            valid_agg[m] = MetricsAgg(mean=float(va.mean()), std=float(va.std(ddof=0)))

        best_n = int(statistics.median(best_iters))
        best_n = max(1, min(best_n, int(config.n_estimators_cap)))

        dfull = xgb.DMatrix(full_sd.x, label=full_sd.y, feature_names=list(feature_names))
        final = xgb.train(
            params=best_params,
            dtrain=dfull,
            num_boost_round=best_n,
            evals=[(dfull, 'train')],
            verbose_eval=False,
        )

        gain = final.get_score(importance_type='gain')
        fi: dict[str, float] = {name: float(gain.get(name, 0.0)) for name in feature_names}

        tuning_report = {
            'n_trials': int(n_trials),
            'timeout_sec': timeout_sec,
            'best_value': float(study.best_value),
            'best_params': dict(study.best_params),
        }

        return TrainingReport(
            model=final,
            model_params=dict(best_params),
            training_params={
                'early_stopping_rounds': int(config.early_stopping_rounds),
                'n_estimators_cap': int(config.n_estimators_cap),
                'n_estimators_final': int(best_n),
                'primary_metric': config.primary_metric,
                'metrics': list(config.metrics),
                'tuning_report': tuning_report,
            },
            metrics=MetricsReport(train=train_agg, valid=valid_agg),
            feature_names=tuple(feature_names),
            feature_importance_gain=fi,
        )
