from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tsfit.application.ports import FeatureBuilder
from tsfit.application.types import DatasetHandle, SupervisedDataset
from tsfit.domain.exceptions import ValidationError
from tsfit.domain.value_objects import DatasetSchema, FeaturePlan


_TARGET_COL = 'tsfit_target'


def _ensure_df(ds: DatasetHandle) -> pd.DataFrame:
    df = ds.payload
    if not isinstance(df, pd.DataFrame):
        raise ValidationError('dataset должен быть DataFrame внутри DatasetHandle')
    return df


def _build_features(
    df: pd.DataFrame,
    schema: DatasetSchema,
    plan: FeaturePlan,
) -> tuple[pd.DataFrame, list[str], int]:
    out = df.copy()

    # гарантируем порядок по времени
    out = out.sort_values(schema.timestamp_col, kind='mergesort').reset_index(drop=True)

    feature_names: list[str] = []

    if plan.add_time_index:
        out['t_idx'] = np.arange(len(out), dtype='int64')
        out['t_idx_coarse'] = (out['t_idx'] // 50).astype('int64')
        feature_names.append('t_idx_coarse')

    if plan.add_delta_t:
        ts = out[schema.timestamp_col]
        dt = ts.diff().dt.total_seconds().fillna(0.0)
        out['delta_t'] = dt.astype('float64')
        feature_names.append('delta_t')

    # лаги: создаём с шагом 10 как и было
    y = out[schema.target_col].astype('float64')
    created_lags: list[int] = []
    for i in range(1, plan.max_lag + 1):
        name = f'y_lag_{i}'
        out[name] = y.shift(i)
        feature_names.append(name)
        created_lags.append(i)

    if not created_lags:
        raise ValidationError('не создано ни одного лага: проверь max_lag')

    last_lag_created = max(created_lags)

    if last_lag_created >= 11:
        out['y_diff_10'] = out['y_lag_1'] - out['y_lag_11']
        feature_names.append('y_diff_10')
    if last_lag_created >= 21:
        out['y_diff_20'] = out['y_lag_1'] - out['y_lag_21']
        feature_names.append('y_diff_20')
    if last_lag_created >= 31:
        out['y_diff_30'] = out['y_lag_1'] - out['y_lag_31']
        feature_names.append('y_diff_30')

    # роллинги по прошлому (строго без текущего)
    y1 = y.shift(1)
    for w in plan.rolling_windows:
        name = f'y_roll_mean_{w}'
        out[name] = y1.rolling(window=w, min_periods=w, center=False).mean()
        feature_names.append(name)

        name_med = f'y_roll_median_{w}'
        out[name_med] = y1.rolling(window=w, min_periods=w, center=False).median()
        feature_names.append(name_med)

        name_ewm = f'y_ewm_mean_{w}'
        out[name_ewm] = y1.ewm(span=w, adjust=False, min_periods=w).mean()
        feature_names.append(name_ewm)

    for w in plan.rolling_std_windows:
        name = f'y_roll_std_{w}'
        out[name] = y1.rolling(window=w, min_periods=w, center=False).std(ddof=0)
        feature_names.append(name)

    if plan.add_season_sin_cos and plan.season_period is not None and plan.add_time_index:
        p = float(plan.season_period)
        t = out['t_idx'].astype('float64')
        out['season_sin'] = np.sin(2.0 * math.pi * t / p)
        out['season_cos'] = np.cos(2.0 * math.pi * t / p)
        feature_names.extend(['season_sin', 'season_cos'])

    if plan.add_exp_features:
        base = out['y_lag_1'].astype('float64')
        out['y_log1p_lag_1'] = np.log1p(np.clip(base, 0.0, None))
        feature_names.append('y_log1p_lag_1')

    # экзогены и индикаторы пропусков
    for c in schema.exog_cols:
        feature_names.append(c)
        ind = f'{c}__was_missing'
        if ind in out.columns:
            feature_names.append(ind)

    # приводим к float64 то, что должно быть числом
    for c in feature_names:
        out[c] = pd.to_numeric(out[c], errors='coerce')

    return out, feature_names, last_lag_created


class PandasFeatureBuilder(FeatureBuilder):
    def build_supervised(
        self,
        ds: DatasetHandle,
        schema: DatasetSchema,
        plan: FeaturePlan,
        *,
        horizon: int,
    ) -> DatasetHandle:
        if horizon < 1:
            raise ValidationError('horizon должен быть >= 1')

        df = _ensure_df(ds)
        feats, feature_names, last_lag_created = _build_features(df, schema, plan)

        y = feats[schema.target_col].astype('float64')
        y_fut = y.shift(-horizon)

        # таргет + трансформация
        if plan.add_exp_features:
            feats[_TARGET_COL] = np.log1p(np.clip(y_fut, 0.0, None))
            feats.attrs['target_transform'] = 'log1p'
        else:
            feats[_TARGET_COL] = y_fut
            feats.attrs['target_transform'] = 'identity'

        # убираем строки без целевого (будущего)
        feats = feats.dropna(subset=[_TARGET_COL])

        # убираем строки, где нет полного лагового окна
        last_lag_col = f'y_lag_{last_lag_created}'
        feats = feats.dropna(subset=[last_lag_col])

        keep_cols = [schema.timestamp_col, _TARGET_COL, *feature_names]
        feats = feats[keep_cols].reset_index(drop=True)

        feats.attrs['feature_names'] = feature_names
        feats.attrs['last_lag_created'] = int(last_lag_created)
        return DatasetHandle(payload=feats)

    def split_xy(self, supervised: DatasetHandle, schema: DatasetSchema) -> SupervisedDataset:
        df = _ensure_df(supervised)
        feature_names = df.attrs.get('feature_names')
        if not feature_names:
            feature_names = [c for c in df.columns if c not in (schema.timestamp_col, _TARGET_COL)]

        x = df[list(feature_names)].to_numpy(dtype='float64', copy=False)
        y = df[_TARGET_COL].to_numpy(dtype='float64', copy=False)

        return SupervisedDataset(
            x=x,
            y=y,
            feature_names=tuple(feature_names),
            n_rows=int(x.shape[0]),
            n_features=int(x.shape[1]),
        )

    def build_last_x(
        self,
        ds: DatasetHandle,
        schema: DatasetSchema,
        plan: FeaturePlan,
    ) -> tuple[object, tuple[str, ...]]:
        df = _ensure_df(ds)
        feats, feature_names, last_lag_created = _build_features(df, schema, plan)

        last_lag_col = f'y_lag_{last_lag_created}'
        valid_rows = feats[~feats[last_lag_col].isna()]
        if valid_rows.empty:
            raise ValidationError('недостаточно истории для лагов')

        last_row = valid_rows.iloc[[-1]]
        x = last_row[list(feature_names)].to_numpy(dtype='float64', copy=False)
        return x, tuple(feature_names)
