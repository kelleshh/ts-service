import numpy as np

import pytest

pl = pytest.importorskip('polars')


def test_rolling_mean_and_std_match_lags(schema_two_series, ts_cfg_basic, two_series_frame) -> None:
    from tsfit.infrastructure.dataset.builder_polars import PolarsTimeSeriesDatasetBuilder
    # исходные данные

    df = two_series_frame

    # timestamp в тестовых данных строковый, делаем как в реальном пайплайне (через parser)
    df = df.with_columns(
        pl.col('ts').str.strptime(pl.Datetime(time_unit='us', time_zone='UTC'), strict=False)
    )
    df = df.sort(['sid', 'ts'])

    builder = PolarsTimeSeriesDatasetBuilder()
    built = builder.build_train_valid(df, schema_two_series, ts_cfg_basic)

    cols = built.feature_names
    assert 'lag_1' in cols and 'lag_2' in cols and 'lag_3' in cols
    assert 'rolling_mean_3' in cols
    assert 'rolling_std_3' in cols
    assert 'rolling_min_3' in cols
    assert 'rolling_max_3' in cols
    assert 'diff_1' in cols and 'diff_3' in cols

    # проверяем отсутствие NaN/None
    assert np.isfinite(built.X_train).all()
    assert np.isfinite(built.X_valid).all()
    assert np.isfinite(built.y_train).all()
    assert np.isfinite(built.y_valid).all()

    def col_idx(name: str) -> int:
        return cols.index(name)

    for X in (built.X_train, built.X_valid):
        lag1 = X[:, col_idx('lag_1')]
        lag2 = X[:, col_idx('lag_2')]
        lag3 = X[:, col_idx('lag_3')]
        mean_from_lags = (lag1 + lag2 + lag3) / 3.0
        std_from_lags = np.std(np.stack([lag1, lag2, lag3], axis=1), axis=1, ddof=0)

        assert np.allclose(X[:, col_idx('rolling_mean_3')], mean_from_lags, atol=1e-12)
        assert np.allclose(X[:, col_idx('rolling_std_3')], std_from_lags, atol=1e-12)
