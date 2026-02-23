import sys
from pathlib import Path
from polars import DataFrame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pytest

pl = pytest.importorskip('polars')

from tsfit.domain.value_objects import (
    DatasetSchemaValueObject, 
    TimeSeriesConfigValueObject, 
    FeatureSpecValueObject, 
    SplitConfigValueObject, 
    TrainingConfigValueObject, 
    TuningConfigValueObject
)


@pytest.fixture()
def schema_two_series() -> DatasetSchemaValueObject:
    return DatasetSchemaValueObject(
        timestamp_col='ts',
        target_col='y',
        series_id_col='sid',
        exogenous_cols=[],
    )


@pytest.fixture()
def ts_cfg_basic() -> TimeSeriesConfigValueObject:
    return TimeSeriesConfigValueObject(
        horizon=1,
        features=FeatureSpecValueObject(
            lags=[1, 2, 3],
            rolling_mean_windows=[3],
            rolling_std_windows=[3],
            rolling_min_windows=[3],
            rolling_max_windows=[3],
            diff_lags=[1, 3],
        ),
        split=SplitConfigValueObject(valid_fraction=0.2, min_valid_size=3),
    )


@pytest.fixture()
def training_cfg_no_tuning() -> TrainingConfigValueObject:
    return TrainingConfigValueObject(
        xgb_params={'n_estimators': 50},
        metrics=['rmse', 'mae'],
        primary_metric='rmse',
        tuning=TuningConfigValueObject(enabled=False),
    )


def make_two_series_frame(n: int = 30) -> DataFrame:
    rows = []
    for sid in ['A', 'B']:
        for i in range(n):
            rows.append(
                {
                    'sid': sid,
                    'ts': f'2024-01-{i+1:02d}',
                    'y': float(i + 1),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture()
def two_series_frame() -> DataFrame:
    '''
    Небольшой датасет из двух рядов для тестов инфраструктуры
    '''
    return make_two_series_frame(n=30)