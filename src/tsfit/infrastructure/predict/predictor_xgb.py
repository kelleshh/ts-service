from __future__ import annotations

import numpy as np
import xgboost as xgb

from tsfit.application.ports import Predictor
from tsfit.domain.exceptions import ValidationError


# Adapter (Infrastructure)
class XGBoostPredictor(Predictor):
    def predict_one(self, model: object, x: object) -> float:
        if not isinstance(model, xgb.Booster):
            raise ValidationError('ожидался xgboost.Booster')
        x_arr = np.asarray(x, dtype='float64')
        d = xgb.DMatrix(x_arr)
        pred = model.predict(d)
        return float(pred.reshape(-1)[0])
