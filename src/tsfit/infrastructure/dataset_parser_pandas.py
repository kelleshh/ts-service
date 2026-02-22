from typing import Any
import pandas as pd

from tsfit.application.ports import DatasetParser
from tsfit.domain.spec import DatasetSchema
from tsfit.domain.errors import ValidationError


class PandasDatasetParser(DatasetParser):
    '''
    Парсер датасета (временного ряда):
    - using pandas
    1) timestamp парсится в datetime
    2) проверка пропусков в таргете
    3) сортировка по времени
    4) проверка на дубли
    '''

    def parse(self, rows: list[dict[str, Any]], schema: DatasetSchema) -> pd.DataFrame:
        if not rows:
            raise ValidationError('dataset пустой')

        df = pd.DataFrame(rows)

        required = [schema.timestamp_col, schema.target_col, *schema.exogenous_cols]
        if schema.series_id_col:
            required.append(schema.series_id_col)

        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValidationError(f'dataset: нет колонок {missing_cols}')

        # timestamp -> datetime
        ts_col = schema.timestamp_col
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce', utc=True)

        bad_ts = df[df[ts_col].isna()]
        if not bad_ts.empty:
            idxs = bad_ts.index[:5].tolist()
            raise ValidationError(
                f'timestamp не парсится в datetime. Примеры строк: {idxs}'
                )

        # target не должен быть NaN (иначе лаги/цель будут мусор)
        tgt = schema.target_col
        if df[tgt].isna().any():
            idxs = df[df[tgt].isna()].index[:5].tolist()
            raise ValidationError(
                f'target содержит пропуски: {idxs}'
                )

        # сортировка
        sort_cols = [ts_col]
        if schema.series_id_col:
            sort_cols = [schema.series_id_col, ts_col]

        df = df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

        # проверка дублей времени внутри серии
        if schema.series_id_col:
            dup_mask = df.duplicated(subset=[schema.series_id_col, ts_col], keep=False)
        else:
            dup_mask = df.duplicated(subset=[ts_col], keep=False)

        if dup_mask.any():
            idxs = df[dup_mask].index[:10].tolist()
            raise ValidationError(f'Есть дубли timestamp: {idxs}')

        return df