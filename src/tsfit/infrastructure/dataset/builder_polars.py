from __future__ import annotations

from typing import Sequence

import polars as pl
import numpy as np

from tsfit.application.ports import BuiltDataset, Frame, TimeSeriesDatasetBuilder
from tsfit.domain.value_objects import DatasetSchemaValueObject, TimeSeriesConfigValueObject
from tsfit.domain.exceptions import ValidationError
from tsfit.infrastructure.preprocess.polars_frame import preprocess_frame_for_training

# хелперы для проверки распределения

def _is_exponential_series(y: np.ndarray) -> bool:
    '''
    Простая эвристика: ряд похож на экспоненту, если
    1) отношения соседних значений y[t] / y[t-1] почти постоянны
    2) в логарифмах ряд почти линейный (log(y) ~~ a + b*t)

    Это нужно чтобы включать дополнительные преобразования признаков там где они чаще всего дают пользу
    '''

    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    # слишком короткий ряд => не делаем выводов
    if y.size < 12:
        return False

    # логарифм требует положительности. Если есть отрицательные/нули, то
    # сдвигаем весь ряд вверх (одинаково для всех значений).
    mn = float(np.min(y))
    if not np.isfinite(mn):
        return False

    eps = 1e-9
    if mn <= 0.0:
        y = y - mn + eps

    # отношения соседних точек: для чистой экспоненты они константны
    prev = y[:-1]
    curr = y[1:]
    good = (prev > 0.0) & np.isfinite(prev) & np.isfinite(curr)
    if int(np.sum(good)) < 10:
        return False

    ratios = curr[good] / prev[good]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size < 10:
        return False

    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios, ddof=0))

    # если средний множитель почти 1 то это не экспонента, а почти константа
    if abs(mean_r - 1.0) < 0.01:
        return False

    cv = std_r / (abs(mean_r) + eps)

    # дополнительно: логарифм ряда должен хорошо объясняться прямой
    logy = np.log(y)
    t = np.arange(logy.size, dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, logy, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((logy - pred) ** 2))
    ss_tot = float(np.sum((logy - float(np.mean(logy))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    # (эвристика) пороги подобраны так, чтобы линейные/полиномиальные тренды обычно не попадали в "экспоненциальные"
    return (cv <= 0.08) and (r2 >= 0.93)


def _should_add_exp_features(df: pl.DataFrame, targ_col: str, sid_col: str | None) -> bool:
    '''
    Решение на уровне всего датасета: добавлять ли лог/эксп преобразования

    Если рядов несколько (есть series_id), то достаточно чтобы хотя бы один ряд
    был похож на экспоненту
    '''

    if sid_col:
        grouped = (
            df.group_by(sid_col)
            .agg(pl.col(targ_col).cast(pl.Float64).alias('_y'))
        )
        for row in grouped.iter_rows(named=True):
            y_list = row.get('_y')
            if y_list is None:
                continue
            if _is_exponential_series(np.asarray(y_list, dtype=float)):
                return True
        return False

    y = df.select(pl.col(targ_col).cast(pl.Float64)).to_series().to_numpy()
    return _is_exponential_series(y)


def _signed_log1p(expr: pl.Expr) -> pl.Expr:
    abs_x = expr.abs()
    # sign(0) = 0 => 0 * log1p(0) = 0
    return pl.when(expr >= 0).then(abs_x.log1p()).otherwise(-abs_x.log1p())


def _exp_scaled(expr: pl.Expr, scale: float) -> pl.Expr:
    # нормируем и ограничиваем
    s = float(scale) if np.isfinite(scale) and float(scale) > 0.0 else 1.0
    z = expr / pl.lit(s)
    z = (
        pl.when(z > pl.lit(10.0)).then(pl.lit(10.0))
        .when(z < pl.lit(-10.0)).then(pl.lit(-10.0))
        .otherwise(z)
    )
    return z.exp()





class PolarsTimeSeriesDatasetBuilder(TimeSeriesDatasetBuilder):
    '''
    Превращает временной ряд в supervised-таблицу (вида X, y)
    '''

    def build_train_valid(self, 
                          frame: pl.DataFrame, 
                          schema: DatasetSchemaValueObject, 
                          cfg: TimeSeriesConfigValueObject) -> BuiltDataset:
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
        n_rows_raw = int(df.height) # сырые строки сразу считаются до препроцессинга


        # ожидается что timestamp уже стал datetime
        if df.schema.get(ts_col) not in (pl.Datetime, pl.Datetime(time_zone='UTC'), pl.Datetime(time_unit='us', time_zone='UTC')):
            try:
                _ = df.select(pl.col(ts_col).dt.year()).head(1)
            except Exception as e:
                raise ValidationError(f'timestamp_col должен быть Datetime (после парсинга). Ошибка: {e}')

        # препроцессинг: чистка y/x и ffill экзогенов
        df = preprocess_frame_for_training(
            df=df,
            ts_col=ts_col,
            target_col=targ_col,
            exog_cols=schema.exogenous_cols,
            series_id_col=sid_col,
        )
            

        # разметка сырой валидации
        if sid_col:
            # колонки индексов внутри серии и длина серии
            df = df.with_columns(
                (pl.col(ts_col).cum_count().over(sid_col) - pl.lit(1)).cast(pl.Int64).alias('_row_in_series'),
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
        
        # решение - добавлять ли лог/эксп признаки
        use_exp_features = _should_add_exp_features(df, targ_col=targ_col, sid_col=sid_col)

        # масштаб для экспрненциальных преобразований
        scale_for_exp = 1.0
        if use_exp_features:
            y_abs = df.select(pl.col(targ_col).cast(pl.Float64).abs().median()).item()
            if isinstance(y_abs, (int, float)) and float(y_abs) > 0.0 and np.isfinite(float(y_abs)):
                scale_for_exp = float(y_abs)

        # таргет y(t+h) и метка valid по будущему
        if sid_col:
            df = df.with_columns(
                pl.col(targ_col).shift(-h).over(sid_col).alias('_y'),
                pl.col('_is_valid_raw')
                .shift(-h)
                .over(sid_col)
                .fill_null(False)
                .cast(pl.Boolean)
                .alias('_is_valid_example'),
            )
        else:
            df = df.with_columns(
                pl.col(targ_col).shift(-h).alias('_y'),
                pl.col('_is_valid_raw').shift(-h).fill_null(False).cast(pl.Boolean).alias(
                    '_is_valid_example'
                ),
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

        
        # логарифмовые/экспоненциальные преобразования признаков
        # добавляем только если ряд выглядит как экспонента
        if use_exp_features:
            base_for_transform = [
                c
                for c in feature_cols
                if c.startswith('lag_')
                or c.startswith('diff_')
                or c.startswith('rolling_mean_')
                or c.startswith('rolling_std_')
                or c.startswith('rolling_min_')
                or c.startswith('rolling_max_')
            ]

            extra_cols: list[str] = []
            exprs: list[pl.Expr] = []
            for base_col in base_for_transform:
                log_name = f'{base_col}_log1p'
                exp_name = f'{base_col}_exp'

                exprs.append(_signed_log1p(pl.col(base_col)).alias(log_name))
                exprs.append(_exp_scaled(pl.col(base_col), scale_for_exp).alias(exp_name))
                extra_cols.extend([log_name, exp_name])

            df = df.with_columns(exprs)
            feature_cols.extend(extra_cols)


        # экзогены + индикаторы пропусков
        for col in schema.exogenous_cols:
            feature_cols.append(col)
            feature_cols.append(f'{col}__is_missing')


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


        # фильтрация строк где нельзя построить supervised пример.
        # экзогены могут оставаться null (например в самом начале ряда).
        # xgboost умеет missing (nan) и это не считается утечкой.
        nullable_cols = set(schema.exogenous_cols)
        needed = ['_y', *[c for c in feature_cols if c not in nullable_cols]]
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

            n_rows_raw=n_rows_raw,
            n_total_after_features=int(df2.height),
            n_train=int(train_df.height),
            n_valid=int(valid_df.height),
            n_features=int(len(feature_cols)),
        )
