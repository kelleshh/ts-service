from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from tsfit.application.ports import ModelTrainer


class XGBModelTrainer(ModelTrainer):
    def train_and_eval(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        params: dict[str, Any],
    ) -> dict[str, Any]:

        user_params = dict(params or {})

        # базовые гиперы # TODO: сделать через optuna!
        defaults: dict[str, Any] = {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        merged = {**defaults, **user_params}

        model = XGBRegressor(**merged)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        pred = model.predict(X_valid)

        rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
        mae = float(mean_absolute_error(y_valid, pred))

        metrics = {"rmse": rmse, "mae": mae}

        # importance (топ-50 фичей)
        feature_importance: dict[str, float] = {}
        if hasattr(model, "feature_importances_") and hasattr(X_train, "columns"):
            fi = model.feature_importances_
            feature_importance = {
                str(name): float(val) for name, val in zip(list(X_train.columns), fi)
            }
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:50]
            )

        return {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model_params": merged,
        }