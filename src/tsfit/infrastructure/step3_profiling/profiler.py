from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tsfit.application.ports import SeriesProfiler, TargetInspector
from tsfit.application.types import DatasetHandle
from tsfit.domain.exceptions import ValidationError
from tsfit.domain.value_objects import DatasetSchema, SeriesProfile


def _regularity(ts: pd.Series) -> bool:
    if len(ts) < 3:
        return True
    deltas = ts.sort_values().diff().dropna()
    if deltas.empty:
        return True

    # приводим к int наносекунд для стабильности
    if pd.api.types.is_timedelta64_dtype(deltas):
        v = deltas.dt.total_seconds().astype('float64')
    else:
        v = deltas.astype('float64')

    counts = v.value_counts(dropna=True)
    if counts.empty:
        return True

    share = float(counts.iloc[0]) / float(len(v))
    return share >= 0.95


def _trend(y: pd.Series) -> bool:
    if len(y) < 10:
        return False
    t = pd.Series(np.arange(len(y), dtype='float64'))
    c = t.corr(y.astype('float64'))
    if c is None or np.isnan(c):
        return False
    return abs(float(c)) >= 0.3


def _best_season_period(y: pd.Series) -> int | None:
    # простая эвристика: ищем сильную автокорреляцию на фиксированных лаговых периодах
    n = len(y)
    if n < 40:
        return None

    yv = y.astype('float64')
    candidates = [7, 12, 24, 30]
    best_p: int | None = None
    best_corr = 0.0
    for p in candidates:
        if p <= 1 or n <= p * 3:
            continue
        a = yv.iloc[p:]
        b = yv.iloc[:-p]
        c = a.corr(b)
        if c is None or np.isnan(c):
            continue
        c = float(c)
        if c > best_corr:
            best_corr = c
            best_p = p

    if best_p is None:
        return None
    if best_corr < 0.3:
        return None
    return best_p


def _is_exponential(y: pd.Series) -> bool:
    if len(y) < 30:
        return False
    # экспоненциальность здесь трактуем как сильную правую асимметрию при положительных значениях
    yv = y.astype('float64')
    pos_share = float((yv > 0).mean())
    if pos_share < 0.95:
        return False

    skew = float(yv.skew()) if not np.isnan(yv.skew()) else 0.0  # type: ignore
    if skew < 1.0:
        return False
    
    logv = np.log1p(yv)

    skew_log = float(pd.Series(logv).skew()) if not np.isnan(pd.Series(logv).skew()) else skew # может все обернуть в сериес?
    return skew_log <= 0.7 * skew


# Adapter (Infrastructure)
class DefaultSeriesProfiler(SeriesProfiler, TargetInspector):
    def profile(self, ds: DatasetHandle, schema: DatasetSchema, *, min_train_ratio: float) -> SeriesProfile:
        df = ds.payload
        if not isinstance(df, pd.DataFrame):
            raise ValidationError('dataset должен быть DataFrame внутри DatasetHandle')
        if schema.timestamp_col not in df.columns or schema.target_col not in df.columns:
            raise ValidationError('нет обязательных колонок')

        n_train = int(len(df))
        if n_train <= 0:
            raise ValidationError('после препроцессинга нет данных')

        is_regular = _regularity(df[schema.timestamp_col])

        w = max(5, int(math.ceil(min_train_ratio * n_train)))
        w = min(w, n_train)
        y_win = df[schema.target_col].iloc[:w]

        has_trend = _trend(y_win)
        season_period = _best_season_period(y_win) if is_regular else None
        is_exp = _is_exponential(y_win)

        return SeriesProfile(
            n_train=n_train,
            is_regular=is_regular,
            has_trend=has_trend,
            season_period=season_period,
            is_exponential=is_exp,
        )

    def frac_abs_y_lt_eps(self, ds: DatasetHandle, schema: DatasetSchema, *, eps: float) -> float:
        df = ds.payload
        if not isinstance(df, pd.DataFrame):
            raise ValidationError('dataset должен быть DataFrame внутри DatasetHandle')
        if schema.target_col not in df.columns:
            raise ValidationError('нет target_col')
        y = df[schema.target_col].astype('float64')
        if len(y) == 0:
            return 1.0
        frac = float((y.abs() < float(eps)).mean())
        return frac
