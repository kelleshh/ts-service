import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pytest

from tsfit.infrastructure.dataset_parser_pandas import PandasDatasetParser
from tsfit.domain.spec import DatasetSchema
from tsfit.domain.errors import ValidationError


def test_parser_rejects_duplicate_timestamps_within_series() -> None:
    schema = DatasetSchema(
        timestamp_col='ts',
        target_col='y',
        series_id_col='sid',
        exogenous_cols=[],
    )
    rows = [
        {'sid': 'A', 'ts': '2024-01-01', 'y': 1.0},
        {'sid': 'A', 'ts': '2024-01-01', 'y': 2.0},  # дубль времени в серии A
        {'sid': 'B', 'ts': '2024-01-01', 'y': 3.0},
    ]

    parser = PandasDatasetParser()
    with pytest.raises(ValidationError):
        parser.parse(rows, schema)

