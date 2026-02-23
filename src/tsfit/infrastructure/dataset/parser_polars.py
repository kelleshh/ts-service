from __future__ import annotations

from typing import Any
import re
from datetime import datetime, timedelta, timezone

import polars as pl

from tsfit.application.ports import DatasetParser
from tsfit.domain.value_objects import DatasetSchemaValueObject
from tsfit.domain.exceptions import ValidationError

_RE_YMD = re.compile(r'^\s*(\d{4})-(\d{2})-(\d{2})\s*$')


def _parse_ts_any(v: Any) -> datetime | None:
    '''
    Приводит timestamp к datetime
    Поддерживает:
    - int/float epoch (s/ms/ns эвристикой)
    - строки ISO (в т.ч. '...Z')
    - строки YYYY-MM-DD с переппонением дня (2024-01-32 -> 2024-02-01)
    - числовые строки ('0', '6700000000') как эпохи
    '''
    if v is None:
        return None

    # epoch числа
    if isinstance(v, (int, float)):
        x = float(v)
        # эвристика единиц по порядку величины
        if abs(x) >= 1e14:      # похоже на ns
            sec = x / 1e9
        elif abs(x) >= 1e12:    # похоже на ms
            sec = x / 1e3
        else:                   # s
            sec = x
        try:
            return datetime.utcfromtimestamp(sec)
        except (OverflowError, OSError, ValueError):
            return None

    # строки
    if isinstance(v, str):
        s = v.strip()

        # числовая строка -> epoch
        if s and (s.isdigit() or (s.startswith('-') and s[1:].isdigit())):
            try:
                return _parse_ts_any(int(s))
            except ValueError:
                return None

        # ISO
        s_iso = s.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(s_iso)
            # приводим к naive UTC
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            pass

        # YYYY-MM-DD
        m = _RE_YMD.match(s)
        if m:
            y, mo, d = map(int, m.groups())
            try:
                base = datetime(y, mo, 1)
            except ValueError:
                return None
            return base + timedelta(days=d - 1)

    return None


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
              schema: DatasetSchemaValueObject) -> pl.DataFrame:
        
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

        # сохраняем исходное значение timestamp для сообщений об ошибках
        df = df.with_row_index(name='_row')
        df = df.with_columns(pl.col(ts).alias('_ts_raw'))

        ts_dtype = df.schema.get(ts)

        if ts_dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        ):
            # эвристика по порядку величины (epoch seconds / ms / ns)
            max_ts = df.select(pl.col(ts).max()).item()
            unit = 's'
            if isinstance(max_ts, (int, float)):
                if max_ts >= 10**14:
                    unit = 'ns'
                elif max_ts >= 10**12:
                    unit = 'ms'
                else:
                    unit = 's'

            df = df.with_columns(
                pl.from_epoch(pl.col(ts).cast(pl.Int64), time_unit=unit)
                .dt.replace_time_zone('UTC')
                .alias(ts)
            )
        else:
            df = df.with_columns(
                pl.col(ts)
                .map_elements(_parse_ts_any, return_dtype=pl.Datetime(time_unit='us'))
                .dt.replace_time_zone('UTC')
                .alias(ts)
            )

        bad_ts = df.filter(pl.col(ts).is_null())
        if bad_ts.height > 0:
            examples = bad_ts.select(['_row', '_ts_raw']).head(5).to_dicts()
            raise ValidationError(f'timestamp не парсится в datetime (первые 5): {examples}')

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