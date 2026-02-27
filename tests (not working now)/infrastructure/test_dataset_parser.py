import pytest

pytest.importorskip('polars')

from tsfit.infrastructure.dataset.parser_polars import PolarsDatasetParser
from tsfit.domain.exceptions import ValidationError


def test_parser_rejects_duplicate_timestamps_within_series(schema_two_series) -> None:
    rows = [
        {'sid': 'A', 'ts': '2024-01-01', 'y': 1.0},
        {'sid': 'A', 'ts': '2024-01-01', 'y': 2.0},  # допустим дубль времени в серии A
        {'sid': 'B', 'ts': '2024-01-01', 'y': 3.0},
    ]

    parser = PolarsDatasetParser()
    with pytest.raises(ValidationError):
        parser.parse(rows, schema_two_series)
