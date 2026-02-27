from __future__ import annotations

import pandas as pd

from tsfit.application.ports import DatasetParser
from tsfit.application.types import DatasetHandle


# Adapter (Infrastructure)
class PandasDatasetParser(DatasetParser):
    def parse(self, rows: list[dict]) -> DatasetHandle:
        df = pd.DataFrame(rows)
        return DatasetHandle(payload=df)
