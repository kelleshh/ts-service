from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from tsfit.application.ports import ModelTuner
from tsfit.application.types import SupervisedDataset
from tsfit.application.results import MetricsAgg, MetricsReport, TrainingReport
from tsfit.domain.value_objects import CVPlan, DatasetSchema, FeaturePlan, TrainingConfig
from tsfit.infrastructure.step4_feature_generation.feature_builder import PandasFeatureBuilder
from tsfit.infrastructure.step5_cross_validation.walk_forward import DefaultWalkForwardSplitter
from tsfit.infrastructure.step7_training.xgb_trainer import _normalize_params, _prepare_eval, _time_weights
from tsfit.infrastructure.step8_evaluation.metrics import compute_all


_SMALL_DATA_MAX_ROWS = 120


@dataclass(frozen=True, slots=True)
class _LrNEstimatorsChoice:
    eta: float
    n_estimators: int
    best_value: float


def _user_specified_eta(model_params: dict[str, object]) -> float | None:
    if 'eta' in model_params:
        return float(model_params['eta'])  # type: ignore
    if 'learning_rate' in model_params:
        return float(model_params['learning_rate'])  # type: ignore
    return None


def _default_eta_grid() -> list[float]:
    return [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]


def _extract_valid_curve(evals_result: dict, primary_metric: str) -> np.ndarray:
    valid = evals_result.get('valid')
    if not isinstance(valid, dict) or not valid:
        raise RuntimeError('xgboost не вернул историю valid-метрики')

    if primary_metric in valid:
        v = valid[primary_metric]
    else:
        # иногда имя метрики отличается, в этом случае берем первую
        k0 = next(iter(valid.keys()))
        v = valid[k0]

    return np.asarray(v, dtype='float64')


def _pick_lr_and_n_estimators_by_curve(
    *,
    folds: list[tuple[SupervisedDataset, SupervisedDataset]],
    feature_names: tuple[str, ...],
    base_params: dict[str, object],
    primary_metric: str,
    mape_eps: float,
    eta_grid: list[float],
    n_estimators_cap: int,
) -> _LrNEstimatorsChoice:
    if n_estimators_cap < 2:
        raise RuntimeError('n_estimators_cap должен быть >= 2')

    eval_update, custom_metric, maximize = _prepare_eval(primary_metric, mape_eps=mape_eps)

    best: _LrNEstimatorsChoice | None = None

    for eta in eta_grid:
        params = dict(base_params)
        params.update(eval_update)
        params['eta'] = float(eta)

        curves: list[np.ndarray] = []
        for train_sd, valid_sd in folds:
            w_train = _time_weights(int(train_sd.n_rows))

            dtrain = xgb.DMatrix(train_sd.x, label=train_sd.y, feature_names=list(feature_names), weight=w_train)
            dvalid = xgb.DMatrix(valid_sd.x, label=valid_sd.y, feature_names=list(feature_names))

            evals_result: dict = {}
            xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=int(n_estimators_cap),
                evals=[(dvalid, 'valid')],
                evals_result=evals_result,
                verbose_eval=False,
                custom_metric=custom_metric,
                maximize=maximize,
            )
            curves.append(_extract_valid_curve(evals_result, primary_metric))

        min_len = min(int(c.shape[0]) for c in curves)
        mean_curve = np.mean(np.vstack([c[:min_len] for c in curves]), axis=0)

        best_it = int(np.argmin(mean_curve)) + 1
        best_val = float(mean_curve[best_it - 1])

        cand = _LrNEstimatorsChoice(eta=float(eta), n_estimators=int(best_it), best_value=float(best_val))
        if best is None or cand.best_value < best.best_value:
            best = cand

    if best is None:
        raise RuntimeError('не удалось подобрать learning rate / n_estimators')

    best_n = max(1, min(best.n_estimators, int(n_estimators_cap)))
    return _LrNEstimatorsChoice(eta=float(best.eta), n_estimators=int(best_n), best_value=float(best.best_value))


def _aggregate_metrics(values: list[dict[str, float]], metrics: tuple[str, ...]) -> dict[str, MetricsAgg]:
    out: dict[str, MetricsAgg] = {}
    for m in metrics:
        arr = np.array([d[m] for d in values], dtype='float64')
        out[m] = MetricsAgg(mean=float(arr.mean()), std=float(arr.std(ddof=0)))
    return out


# Adapter (Infrastructure)
class OptunaTuner(ModelTuner):
    def __init__(self) -> None:
        self._splitter = DefaultWalkForwardSplitter()
        self._fb = PandasFeatureBuilder()

    def tune_and_train(
        self,
        supervised,
        schema: DatasetSchema,
        feature_plan: FeaturePlan,  # feature_plan/horizon сохраняем в сигнатуре порта
        cv_plan: CVPlan,
        config: TrainingConfig,
        *,
        horizon: int,
        n_trials: int,
        timeout_sec: int | None,
    ) -> TrainingReport:
        try:
            import optuna
        except Exception as e:
            raise RuntimeError('optuna не установлен') from e

        full_sd = self._fb.split_xy(supervised, schema)
        folds_h = self._splitter.split(supervised, cv_plan)
        folds = [(self._fb.split_xy(f.train, schema), self._fb.split_xy(f.valid, schema)) for f in folds_h]

        feature_names = full_sd.feature_names
        base = _normalize_params(config.model_params)

        is_small_data = int(full_sd.n_rows) <= _SMALL_DATA_MAX_ROWS

        chosen: _LrNEstimatorsChoice | None = None
        fixed_n_estimators: int | None = None
        eta_grid_for_report: list[float] | None = None

        if is_small_data:
            user_eta = _user_specified_eta(config.model_params)
            eta_grid = [float(user_eta)] if user_eta is not None else _default_eta_grid()
            eta_grid_for_report = [float(x) for x in eta_grid]

            chosen = _pick_lr_and_n_estimators_by_curve(
                folds=folds,
                feature_names=feature_names,
                base_params=base,
                primary_metric=config.primary_metric,
                mape_eps=config.mape_eps,
                eta_grid=eta_grid,
                n_estimators_cap=int(config.n_estimators_cap),
            )
            base['eta'] = float(chosen.eta)
            fixed_n_estimators = int(chosen.n_estimators)

        def objective(trial: 'optuna.Trial') -> float:
            params = dict(base)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 6)
            params['min_child_weight'] = trial.suggest_float('min_child_weight', 1.0, 30.0, log=True)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.3, 1.0)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-2, 5.0, log=True)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True)
            params['gamma'] = trial.suggest_float('gamma', 0.0, 5.0)
            params['max_delta_step'] = trial.suggest_float('max_delta_step', 0.0, 10.0)

            # eta подбираем только на больших данных, на малых - он уже выбран отдельно
            if not is_small_data:
                params['eta'] = trial.suggest_float('eta', 0.01, 0.1, log=True)

            eval_update, custom_metric, maximize = _prepare_eval(config.primary_metric, mape_eps=config.mape_eps)
            params.update(eval_update)

            vals: list[float] = []
            for train_sd, valid_sd in folds:
                w_train = _time_weights(int(train_sd.n_rows))

                dtrain = xgb.DMatrix(train_sd.x, label=train_sd.y, feature_names=list(feature_names), weight=w_train)
                dvalid = xgb.DMatrix(valid_sd.x, label=valid_sd.y, feature_names=list(feature_names))

                if fixed_n_estimators is not None:
                    booster = xgb.train(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=int(fixed_n_estimators),
                        evals=[],
                        verbose_eval=False,
                    )
                    yhat = booster.predict(dvalid)
                else:
                    booster = xgb.train(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=int(config.n_estimators_cap),
                        evals=[(dvalid, 'valid')],
                        early_stopping_rounds=int(config.early_stopping_rounds),
                        custom_metric=custom_metric,
                        verbose_eval=False,
                        maximize=maximize,
                    )
                    best_it = int(getattr(booster, 'best_iteration', config.n_estimators_cap - 1))
                    yhat = booster.predict(dvalid, iteration_range=(0, best_it + 1))

                m = compute_all(
                    metrics=[config.primary_metric],
                    y_true=valid_sd.y,  # type: ignore[arg-type]
                    y_pred=yhat,
                    eps=config.mape_eps,
                )[config.primary_metric]
                vals.append(float(m))

            return float(np.mean(vals))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=int(n_trials), timeout=timeout_sec)

        best_params = dict(base)
        best_params.update(study.best_params)

        eval_update, custom_metric, maximize = _prepare_eval(config.primary_metric, mape_eps=config.mape_eps)
        best_params.update(eval_update)

        # финальный прогон по фолдам (для отчета)
        fold_train_metrics: list[dict[str, float]] = []
        fold_valid_metrics: list[dict[str, float]] = []
        best_iters: list[int] = []

        for train_sd, valid_sd in folds:
            w_train = _time_weights(int(train_sd.n_rows))

            dtrain = xgb.DMatrix(train_sd.x, label=train_sd.y, feature_names=list(feature_names), weight=w_train)
            dvalid = xgb.DMatrix(valid_sd.x, label=valid_sd.y, feature_names=list(feature_names))

            if fixed_n_estimators is not None:
                booster = xgb.train(
                    params=best_params,
                    dtrain=dtrain,
                    num_boost_round=int(fixed_n_estimators),
                    evals=[],
                    verbose_eval=False,
                )
                best_it = int(fixed_n_estimators) - 1
            else:
                booster = xgb.train(
                    params=best_params,
                    dtrain=dtrain,
                    num_boost_round=int(config.n_estimators_cap),
                    evals=[(dtrain, 'train'), (dvalid, 'valid')],
                    early_stopping_rounds=int(config.early_stopping_rounds),
                    custom_metric=custom_metric,
                    verbose_eval=False,
                    maximize=maximize,
                )
                best_it = int(getattr(booster, 'best_iteration', config.n_estimators_cap - 1))

            best_iters.append(best_it + 1)

            yhat_train = booster.predict(dtrain, iteration_range=(0, best_it + 1))
            yhat_valid = booster.predict(dvalid, iteration_range=(0, best_it + 1))

            tail = int(valid_sd.n_rows)
            y_true_train = train_sd.y[-tail:]  # type: ignore
            y_pred_train = yhat_train[-tail:]

            fold_train_metrics.append(
                compute_all(metrics=list(config.metrics), y_true=y_true_train, y_pred=y_pred_train, eps=config.mape_eps)
            )
            fold_valid_metrics.append(
                compute_all(metrics=list(config.metrics), y_true=valid_sd.y, y_pred=yhat_valid, eps=config.mape_eps)  # type: ignore[arg-type]
            )

        train_agg = _aggregate_metrics(fold_train_metrics, tuple(config.metrics))
        valid_agg = _aggregate_metrics(fold_valid_metrics, tuple(config.metrics))

        best_n = int(statistics.median(best_iters))
        best_n = max(1, min(best_n, int(config.n_estimators_cap)))
        if fixed_n_estimators is not None:
            best_n = int(fixed_n_estimators)

        w_full = _time_weights(int(full_sd.n_rows))
        dfull = xgb.DMatrix(full_sd.x, label=full_sd.y, feature_names=list(feature_names), weight=w_full)
        final = xgb.train(
            params=best_params,
            dtrain=dfull,
            num_boost_round=int(best_n),
            evals=[(dfull, 'train')],
            verbose_eval=False,
        )
        final.set_attr(tsfit_target_transform='identity')

        gain = final.get_score(importance_type='gain')
        fi: dict[str, float] = {name: float(gain.get(name, 0.0)) for name in feature_names}  # type: ignore[arg-type]

        tuning_report: dict[str, object] = {
            'n_trials': int(n_trials),
            'timeout_sec': timeout_sec,
            'best_value': float(study.best_value),
            'best_params': dict(study.best_params),
        }
        if chosen is not None:
            tuning_report['stage1_lr_n_estimators'] = {
                'eta': float(chosen.eta),
                'n_estimators': int(chosen.n_estimators),
                'best_value': float(chosen.best_value),
                'eta_grid': [float(x) for x in (eta_grid_for_report or [])],
                'small_data_max_rows': int(_SMALL_DATA_MAX_ROWS),
                'n_rows_supervised': int(full_sd.n_rows),
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
