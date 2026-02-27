from __future__ import annotations

import hashlib
import json

from tsfit.application.ports import PayloadHasher
from tsfit.domain.value_objects import DatasetSchema, TrainingConfig


# Adapter (Infrastructure)
class Sha256PayloadHasher(PayloadHasher):
    def hash_fit_payload(
        self,
        *,
        rows: list[dict],
        schema: DatasetSchema,
        horizon: int,
        training: TrainingConfig,
        tuning: dict[str, object] | None, # чтоб работал для обеих режимов обучения
        policy_version: str,
    ) -> str:
        payload = {
            'rows': rows,
            'schema': {
                'timestamp_col': schema.timestamp_col,
                'target_col': schema.target_col,
                'exog_cols': list(schema.exog_cols),
            },
            'horizon': horizon,
            'training': {
                'primary_metric': training.primary_metric,
                'metrics': list(training.metrics),
                'model_params': dict(training.model_params),
                'early_stopping_rounds': training.early_stopping_rounds,
                'n_estimators_cap': training.n_estimators_cap,
                'mape_eps': training.mape_eps,
                'mape_zero_frac_threshold': training.mape_zero_frac_threshold,
            },
            'tuning': tuning,
            'policy_version': policy_version,
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(',', ':'))
        return hashlib.sha256(s.encode('utf-8')).hexdigest()
