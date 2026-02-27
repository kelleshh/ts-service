from datetime import datetime, timedelta

import numpy as np

import pytest

pl = pytest.importorskip('polars')


def test_exponential_series_adds_log_and_exp_features(schema_two_series, ts_cfg_basic) -> None:
    from tsfit.infrastructure.dataset.builder_polars import PolarsTimeSeriesDatasetBuilder

    rows = []
    base_dt = datetime(2024, 1, 1)
    for i in range(60):
        rows.append(
            {
                'sid': 'A',
                'ts': (base_dt + timedelta(days=i)).strftime('%Y-%m-%d'),
                'y': float(np.exp(0.05 * i)),
            }
        )

    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col('ts').str.strptime(pl.Datetime(time_unit='us', time_zone='UTC'), strict=False)
    )
    df = df.sort(['sid', 'ts'])

    builder = PolarsTimeSeriesDatasetBuilder()
    built = builder.build_train_valid(df, schema_two_series, ts_cfg_basic)

    cols = built.feature_names

    assert 'lag_1' in cols
    assert 'rolling_mean_3' in cols

    assert 'lag_1_log1p' in cols
    assert 'lag_1_exp' in cols
    assert 'rolling_mean_3_log1p' in cols
    assert 'rolling_mean_3_exp' in cols

    assert np.isfinite(built.X_train).all()
    assert np.isfinite(built.X_valid).all()
