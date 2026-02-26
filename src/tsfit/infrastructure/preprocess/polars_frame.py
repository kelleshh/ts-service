from __future__ import annotations

from typing import Iterable

import polars as pl


_BAD_STRINGS = {
    '',
    '-',
    'none',
    'null',
    'nan',
    'inf',
    '+inf',
    '-inf',
}


def _to_float_or_null(col: str) -> pl.Expr:
    # приводим все к строке, чистим мусор, потом пытаемся сделать float
    s = (
        pl.col(col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
    )

    x = (
        pl.when(pl.col(col).is_null())
        .then(None)
        .when(s.is_in(sorted(_BAD_STRINGS)))
        .then(None)
        .otherwise(s.cast(pl.Float64, strict=False))
    )

    # защита от inf/-inf после кастов
    x = pl.when(x.is_infinite()).then(None).otherwise(x)
    return x


def preprocess_frame_for_training(
    *,
    df: pl.DataFrame,
    ts_col: str,
    target_col: str,
    exog_cols: Iterable[str],
    series_id_col: str | None,
) -> pl.DataFrame:
    '''
    Препроцессинг, который обязателен для обучения/инференса.

    Правила:
    - сортировка по времени
    - дубли времени: оставляем последнюю строку (по порядку входа)
    - y: чистим и выкидываем строки, где y отсутствует
    - экзогены: чистим, помечаем пропуски индикаторами, потом ffill строго по прошлому
    '''

    if '_row' not in df.columns:
        df = df.with_row_index('_row')

    sort_cols = [ts_col] if not series_id_col else [series_id_col, ts_col]
    sort_cols_with_row = [*sort_cols, '_row']

    df = df.sort(sort_cols_with_row)

    subset = sort_cols
    df = df.unique(subset=subset, keep='last')
    df = df.sort(sort_cols)

    # чистка y
    df = df.with_columns(_to_float_or_null(target_col).alias(target_col))
    df = df.filter(pl.col(target_col).is_not_null())

    # чистка экзогенов + индикатор пропусков
    exog_cols = list(exog_cols)
    if exog_cols:
        df = df.with_columns([_to_float_or_null(c).alias(c) for c in exog_cols])

        missing_cols = [
            pl.col(c).is_null().cast(pl.Int8).alias(f'{c}__is_missing')
            for c in exog_cols
        ]
        df = df.with_columns(missing_cols)

        # ffill по времени (и внутри серии, если она есть)
        filled = []
        for c in exog_cols:
            e = pl.col(c).fill_null(strategy='forward')
            if series_id_col:
                e = e.over(series_id_col)
            filled.append(e.alias(c))
        df = df.with_columns(filled)

    return df
