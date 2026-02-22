import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
import numpy as np
import pytest

from tsfit.infrastructure.feature_builder_pandas import PandasSupervisedDatasetBuilder
from tsfit.domain.spec import DatasetSchema, TimeSeriesConfig, FeatureSpec, SplitConfig


def _make_frame_two_series(n: int = 20) -> pd.DataFrame:
    # Две серии, по n точек каждая, целевой ряд = 1, 2, ...n (для каждой серии отдельно)
    rows = []
    for sid in ['A', 'B']:
        for i in range(n):
            rows.append(
                {
                    'sid': sid,
                    'ts': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                    'y': float(i + 1),
                }
            )
    return pd.DataFrame(rows)


def test_rolling_features_match_lags_mean_and_std() -> None:
    df = _make_frame_two_series(n=30)

    schema = DatasetSchema(
        timestamp_col='ts',
        target_col='y',
        series_id_col='sid',
        exogenous_cols=[],
    )

    cfg = TimeSeriesConfig(
        horizon=1,
        features=FeatureSpec(
            lags=[1, 2, 3],
            rolling_mean_windows=[3],
            rolling_std_windows=[3],
        ),
        split=SplitConfig(valid_fraction=0.2, min_valid_size=3),
    )

    builder = PandasSupervisedDatasetBuilder()
    X_tr, y_tr, X_va, y_va, cols = builder.build_train_valid(df, schema, cfg)

    assert 'lag_1' in cols and 'lag_2' in cols and 'lag_3' in cols
    assert 'rolling_mean_3' in cols
    assert 'rolling_std_3' in cols

    # Проверка отсутствия NaNов после сборки supervised выборки
    assert not X_tr.isna().any().any()
    assert not X_va.isna().any().any()
    assert not y_tr.isna().any()
    assert not y_va.isna().any()

    # rolling_mean_3 и rolling_std_3 должны совпадать со статистикой по lag_1...lag_3
    for X in (X_tr, X_va):
        mean_from_lags = X[['lag_1', 'lag_2', 'lag_3']].mean(axis=1)
        std_from_lags = X[['lag_1', 'lag_2', 'lag_3']].std(axis=1, ddof=0)

        assert np.allclose(X['rolling_mean_3'].to_numpy(), mean_from_lags.to_numpy(), atol=1e-12)
        assert np.allclose(X['rolling_std_3'].to_numpy(), std_from_lags.to_numpy(), atol=1e-12)
