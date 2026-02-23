from __future__ import annotations

from typing import Any

import polars as pl

from tsfit.application.ports import DatasetParser
from tsfit.domain.spec import DatasetSchema
from tsfit.domain.errors import ValidationError


class PolarsDatasetParser(DatasetParser):
    '''
    Парсер входного JSON в таблицу pl.DataFrame.

    Делает строгие проверки и преобразования:
    1) timestamp парсится в datetime (UTC)
    2) target без пропусков
    3) сортировка по (series_id, timestamp) или по timestamp
    4) запрет дублей timestamp внутри серии (или вообще если series_id нет)
    '''

    def parse(self, 
              rows: list[dict[str, Any]], 
              schema: DatasetSchema) -> pl.DataFrame:
        
        if not rows: raise ValidationError('dataset пустой')

        df = pl.DataFrame(rows)

        # обязательные колонки
        required = [schema.timestamp_col, schema.target_col, *schema.exogenous_cols]
        if schema.series_id_col:
            required.append(schema.series_id_col)

        # проверка на пропущенные обязательные колонки
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValidationError(f'dataset: нет колонок {missing_cols}')
        
        ts = schema.timestamp_col
        targ = schema.target_col
        sid = schema.series_id_col

        # timestamp в datetime (utc)
        df = df.with_columns(
            pl.col(ts)
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime(time_unit='us', time_zone='UTC'), strict=False)
            .alias(ts)
        )

        # проврка timstamp которые не смогли обработать
        bad_ts = df.filter(pl.col(ts).is_null())
        if bad_ts.height > 0:
            idxs = bad_ts.select(pl.int_range(0, pl.len()).alias('_i')).head(5).to_series().to_list()
            raise ValidationError(f'timestamp не парсится в datetime (первые 5): {idxs}')

        # проверка на не null таргета
        if df.filter(pl.col(targ).is_null()).height > 0:
            idxs = df.filter(pl.col(targ).is_null()).select(pl.int_range(0, pl.len()).alias('_i')).head(5)
            raise ValidationError(f'target содержит пропуски (первые 5): {idxs.to_dicts()}')
        
        # сортировка
        to_sort_cols = [ts] if not sid else [sid, ts]
        df = df.sort(to_sort_cols)

        # проверка на дубли timestamp
        if sid:
            dup = (df.group_by([sid, ts])
                   .len()
                   .filter(pl.col('len') > 1))
        else:
            dup = (df.group_by([ts])
                   .len()
                   .filter(pl.col('len') > 1))
        if dup.height > 0:
            raise ValidationError(f'Есть дубли timestamp (первые 10): {dup.head(10).to_dicts()}')

        return df