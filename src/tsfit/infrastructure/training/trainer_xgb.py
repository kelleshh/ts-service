from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor

from tsfit.application.ports import BuiltDataset, ModelTrainer
from tsfit.domain.value_objects import TrainingConfigValueObject
from tsfit.infrastructure.training.xgb_defaults import merge_params

class XGBModelTrainer(ModelTrainer):
    '''
    Инфраструктурная реализация обучателя на базе XGBoost.

    Вход:
    - BuiltDataset (готовые X_train/y_train, X_valid/y_valid)
    - TrainingConfigValueObject (параметры + список метрик)

    Выход:
    - словарь с метриками, параметрами модели и важностью признаков

    '''

    def train_and_eval(self, dataset: BuiltDataset, training: TrainingConfigValueObject) -> dict[str, Any]:

        X_train = dataset.X_train
        y_train = dataset.y_train
        X_valid = dataset.X_valid
        y_valid = dataset.y_valid
        feature_names = dataset.feature_names

        # устанавливает параметры
        params = merge_params(training.model_params)

        model = XGBRegressor(
            **params,
            early_stopping_rounds = 50,
            eval_metric=list(training.metrics))
        
        # метрики считаем и на train, и на valid
        #
        # в evals_result() XGBoost именует наборы как validation_0, validation_1, ...
        # Поэтому порядок важен:
        # validation_0 -> train
        # validation_1 -> valid
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False,
        )

        evals_result = model.evals_result()

        train_name = 'validation_0'
        valid_name = 'validation_1'
        if train_name not in evals_result:
            raise ValueError('XGBoost не вернул evals_result для validation_0 (train)')
        if valid_name not in evals_result:
            raise ValueError('XGBoost не вернул evals_result для validation_1 (valid)')

        train_metrics = evals_result[train_name]
        valid_metrics = evals_result[valid_name]

        metrics: dict[str, float] = {}
        for requested in training.metrics:
            if requested not in valid_metrics:
                known = sorted(valid_metrics.keys())
                raise ValueError(
                    f'XGBoost не вернул метрику {requested} на valid. Доступно: {known}'
                )
            if requested not in train_metrics:
                known = sorted(train_metrics.keys())
                raise ValueError(
                    f'XGBoost не вернул метрику {requested} на train. Доступно: {known}'
                )

            valid_series = valid_metrics[requested]
            train_series = train_metrics[requested]
            if not valid_series:
                raise ValueError(f'Пустая история значений метрики {requested} на valid')
            if not train_series:
                raise ValueError(f'Пустая история значений метрики {requested} на train')

            best_i = getattr(model, 'best_iteration', None)
            idx = int(best_i) if best_i is not None else -1

            # На всякий случай: если best_iteration почему-то выходит за границы
            if idx != -1:
                idx = min(idx, len(valid_series) - 1, len(train_series) - 1)

            # Контракт:
            # - 'rmse' -> valid
            # - 'rmse_train' -> train
            metrics[requested] = float(valid_series[idx])
            metrics[f'{requested}_train'] = float(train_series[idx])

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

        return {
            'metrics': metrics,
            'primary_metric': training.primary_metric,
            'primary_metric_value': float(metrics[training.primary_metric]),
            'feature_importance': feature_importance,
            'model_params': params,
        }
    

    # TODO: почему то метрики ужасные. на среду: надо полностью пересмотреть все что я тут нагородил