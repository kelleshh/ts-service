from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from tsfit.application.ports import BuiltDataset, Frame, TimeSeriesDatasetBuilder
from tsfit.domain.spec import DatasetSchema, TimeSeriesConfig
from tsfit.domain.errors import ValidationError

class PolarsTimeSeriesDatasetBuilder(TimeSeriesDatasetBuilder):
    '''
    Превращает временной ряд в supervised-таблицу (вида X, y)

    Последовательно делает:
    - 
    '''

    def build_train_valid(self, 
                          frame: pl.DataFrame, 
                          schema: DatasetSchema, 
                          cfg: TimeSeriesConfig) -> BuiltDataset:
        ts_col = schema.timestamp_col
        targ_col = schema.target_col
        sid_col = schema.series_id_col

        lags = sorted(set(cfg.features.lags))
        rolling_mean_windows = sorted(set(cfg.features.rolling_mean_windows))
        rolling_std_windows = sorted(set(cfg.features.rolling_std_windows))
        rolling_min_windows = sorted(set(cfg.features.rolling_min_windows))
        rolling_max_windows = sorted(set(cfg.features.rolling_max_windows))
        diff_lags = sorted(set(cfg.features.diff_lags))
        h = int(cfg.horizon) # горизонт

        df = frame.clone()

        # ожидается что timestamp уже стал datetime от парсера
        if df.schema.get(ts_col) not in (pl.Datetime, pl.Datetime(time_zone='UTC'), pl.Datetime(time_unit='us', time_zone='UTC')):
            try:
                _ = df.select(pl.col(ts_col).dt.year()).head(1)
            except Exception as e:
                raise ValidationError(f'timestamp_col должен быть Datetime (после парсинга). Ошибка: {e}')
            
        # разметка сырой валидации
        if sid_col:
            # колонки индексов внутри серии и длина серии
            df = df.with_columns(
                pl.cum_count().over(sid_col).alias('_row_in_series'),
                pl.len().over(sid_col).alias('_len_series'),
            )
            n_valid = (
                (pl.col('_len_series') * pl.lit(cfg.split.valid_fraction))
                .round(0)
                .cast(pl.Int64)
            )
            n_valid = pl.max_horizontal(n_valid, pl.lit(int(cfg.split.min_valid_size)))
            n_valid = pl.min_horizontal(n_valid, pl.col('_len_series') - pl.lit(1))
            df = df.with_columns(
                (pl.col('_row_in_series') >= (pl.col('_len_series') - n_valid)).alias('_is_valid_raw')
            )

            min_len = (
                df.select(pl.col('_len_series').min()).item()
            )

        else:
            n = df.height
            n_valid = int(round(n * cfg.split.valid_fraction))
            n_valid = max(n_valid, int(cfg.split.min_valid_size))
            n_valid = min(n_valid, n - 1)

            df = df.with_columns(
                pl.int_range(0, pl.len()).alias('_row_in_series'),
                pl.lit(n).alias('_len_series'),
                (pl.int_range(0, pl.len()) >= pl.lit(n - n_valid)).alias('_is_valid_raw'),
            )
            min_len = n

        max_lag = int(max(lags))
        max_roll = int(max([*rolling_mean_windows, *rolling_std_windows, *rolling_min_windows, *rolling_max_windows], default=0))
        max_hist = max(max_lag, max_roll, int(max(diff_lags, default=0)))

        if min_len <= h:
            raise ValidationError('Данных недостаточно: min_len <= horizon в одной из серий')
        if max_hist >= (min_len - h):
            raise ValidationError('Слишком большая глубина истории (lags/rolling/diff) для min_len и horizon')

        # таргет y(t+h) и метка valid по будущему
        if sid_col:
            df = df.with_columns(
                pl.col(targ_col).shift(-h).over(sid_col).alias('_y'),
                pl.col('_is_valid_raw').shift(-h).over(sid_col).fill_null(False).cast(pl.Boolean).alias('_is_valid_example'),
            )
        else:
            df = df.with_columns(
                pl.col(targ_col).shift(-h).alias('_y'),
                pl.col('_is_valid_raw').shift(-h).fill_null(False).cast(pl.Boolean).alias('_is_valid_example'),
            )

        feature_cols: list[str] = []

        # лаги
        for lag in lags:
            name = f'lag_{lag}'
            expr = pl.col(targ_col).shift(lag)
            if sid_col:
                expr = expr.over(sid_col)
            df = df.with_columns(expr.alias(name))
            feature_cols.append(name)

        # разности (y(t) - y(t-lag))
        for lag in diff_lags:
            name = f'diff_{lag}'
            base = pl.col(targ_col)
            shifted = pl.col(targ_col).shift(lag)
            if sid_col:
                shifted = shifted.over(sid_col)
            df = df.with_columns((base - shifted).alias(name))
            feature_cols.append(name)

        # rolling-статистики по прошлому (shift(1) чтобы не учитывать текущий y)
        def _rolling(expr: pl.Expr, win: int, fn: str) -> pl.Expr:
            shifted = expr.shift(1)
            if fn == 'mean':
                return shifted.rolling_mean(window_size=win, min_samples=win)
            if fn == 'std':
                return shifted.rolling_std(window_size=win, min_samples=win, ddof=0)
            if fn == 'min':
                return shifted.rolling_min(window_size=win, min_samples=win)
            if fn == 'max':
                return shifted.rolling_max(window_size=win, min_samples=win)
            raise ValueError(fn)

        for win in rolling_mean_windows:
            name = f'rolling_mean_{win}'
            expr = _rolling(pl.col(targ_col), win, 'mean')
            if sid_col:
                expr = expr.over(sid_col)
            df = df.with_columns(expr.alias(name))
            feature_cols.append(name)

        for win in rolling_std_windows:
            name = f'rolling_std_{win}'
            expr = _rolling(pl.col(targ_col), win, 'std')
            if sid_col:
                expr = expr.over(sid_col)
            df = df.with_columns(expr.alias(name))
            feature_cols.append(name)

        for win in rolling_min_windows:
            name = f'rolling_min_{win}'
            expr = _rolling(pl.col(targ_col), win, 'min')
            if sid_col:
                expr = expr.over(sid_col)
            df = df.with_columns(expr.alias(name))
            feature_cols.append(name)

        for win in rolling_max_windows:
            name = f'rolling_max_{win}'
            expr = _rolling(pl.col(targ_col), win, 'max')
            if sid_col:
                expr = expr.over(sid_col)
            df = df.with_columns(expr.alias(name))
            feature_cols.append(name)

        # экзогены
        for col in schema.exogenous_cols:
            feature_cols.append(col)

        # календарные признаки
        df = df.with_columns(
            pl.col(ts_col).dt.weekday().cast(pl.Int16).alias('dow'),
            pl.col(ts_col).dt.month().cast(pl.Int16).alias('month'),
        )
        feature_cols += ['dow', 'month']

        # числовой код серии
        if sid_col:
            # stable mapping: сортирует уникальные значения
            sids = df.select(pl.col(sid_col).unique().sort()).to_series().to_list()
            mapping = {sid: i for i, sid in enumerate(sids)}
            df = df.with_columns(
                pl.col(sid_col).map_elements(mapping.get, return_dtype=pl.Int32).alias('series_code')
            )
            feature_cols.append('series_code')

        # фильтрация строк где нельзя построить supervised пример
        needed = ['_y', *feature_cols]
        not_null_exprs: Sequence[pl.Expr] = [pl.col(c).is_not_null() for c in needed]
        df2 = df.filter(pl.all_horizontal(not_null_exprs))

        if df2.height == 0:
            raise ValidationError('После построения лагов/цели не осталось строк (слишком большие lags/horizon)')

        train_df = df2.filter(~pl.col('_is_valid_example'))
        valid_df = df2.filter(pl.col('_is_valid_example'))

        if train_df.height == 0:
            raise ValidationError('Train пустой')
        if valid_df.height == 0:
            raise ValidationError('Valid пустой')

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df.select('_y').to_numpy().ravel()
        X_valid = valid_df.select(feature_cols).to_numpy()
        y_valid = valid_df.select('_y').to_numpy().ravel()

        return BuiltDataset(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            feature_names=feature_cols,
        )
