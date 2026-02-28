from __future__ import annotations

import numpy as np
import pandas as pd

from tsfit.application.ports import Preprocessor
from tsfit.application.types import DatasetHandle
from tsfit.domain.exceptions import ValidationError
from tsfit.domain.value_objects import DatasetSchema, PreprocessPlan


_BAD_STRINGS = {
    '',
    ' ',
    '-',
    'none',
    'None',
    'null',
    'nan',
    'NaN',
    'inf',
    '-inf',
    'Infinity',
    '-Infinity',
}


def _to_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors='coerce', utc=False, dayfirst=True)


def _clean_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        s2 = s.replace([np.inf, -np.inf], np.nan)
        return s2

    s2 = s.astype('object')
    s2 = s2.replace(list(_BAD_STRINGS), np.nan)
    s2 = s2.replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(s2, errors='coerce')


# Adapter (Infrastructure)
class DefaultPreprocessor(Preprocessor):
    def preprocess(self, ds: DatasetHandle, schema: DatasetSchema, plan: PreprocessPlan) -> DatasetHandle:
        df = ds.payload
        if not isinstance(df, pd.DataFrame):
            raise ValidationError('dataset должен быть DataFrame внутри DatasetHandle')

        missing_cols = [c for c in (schema.timestamp_col, schema.target_col, *schema.exog_cols) if c not in df.columns]
        if missing_cols:
            raise ValidationError(f'в датасете нет колонок: {missing_cols}')

        out = df.copy()

        out[schema.timestamp_col] = _to_datetime_series(out[schema.timestamp_col])
        out = out.dropna(subset=[schema.timestamp_col])
        out = out.sort_values(schema.timestamp_col, kind='mergesort')

        # дубль timestamp: оставляем последнюю строку
        out = out.drop_duplicates(subset=[schema.timestamp_col], keep='last')

        # таргет
        out[schema.target_col] = _clean_numeric(out[schema.target_col])
        if plan.drop_missing_target:
            out = out.dropna(subset=[schema.target_col])

        # экзогены
        for c in schema.exog_cols:
            out[c] = _clean_numeric(out[c])

        if plan.add_missing_indicators:
            for c in schema.exog_cols:
                out[f'{c}__was_missing'] = out[c].isna().astype('int8')

        if plan.ffill_exog and schema.exog_cols:    # заполняем данными из прошлого
            out[list(schema.exog_cols)] = out[list(schema.exog_cols)].ffill()

        out = out.reset_index(drop=True)
        return DatasetHandle(payload=out)
