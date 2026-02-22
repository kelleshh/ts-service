import pandas as pd
import numpy as np

from tsfit.application.ports import SupervisedDatasetBuilder
from tsfit.domain.spec import DatasetSchema, TimeSeriesConfig
from tsfit.domain.errors import ValidationError


class PandasSupervisedDatasetBuilder(SupervisedDatasetBuilder): # возможно стоит это переименовать в более понятное
    '''
    Билдер датасета для обучения модели:
    1) Сначала определяет валидационную часть датасета по времени
    2) Строит признаки (лаги, скользящее среднее и стандартное отклонение) и таргет y(t+h)
    3) Берет train/valid так чтобы в train не попадали примеры, у которых y(t+h) находится в валидационной части
    а в valid попадали
    '''

    def build_train_valid(
        self,
        frame: pd.DataFrame,
        schema: DatasetSchema,
        cfg: TimeSeriesConfig,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:

        ts_col = schema.timestamp_col
        tgt_col = schema.target_col
        sid_col = schema.series_id_col

        lags = sorted(set(cfg.features.lags))
        rolling_mean_windows = sorted(set(cfg.features.rolling_mean_windows))
        rolling_std_windows = sorted(set(cfg.features.rolling_std_windows))
        h = cfg.horizon

        df = frame.copy()

        # размечаем валидационную часть на сыром ряду
        if sid_col:
            df['_row_in_series'] = df.groupby(sid_col).cumcount()
            df['_len_series'] = df.groupby(sid_col)[ts_col].transform('size')

            n = df['_len_series']
            n_valid = (n * cfg.split.valid_fraction).round().astype(int)
            n_valid = np.maximum(n_valid, cfg.split.min_valid_size)
            n_valid = np.minimum(n_valid, n - 1)  # чтобы train не стал пустым

            df['_is_valid_raw'] = df['_row_in_series'] >= (df['_len_series'] - n_valid)

            per_len = df.groupby(sid_col)['_len_series'].max().astype(int)
            min_len = int(per_len.min())
        else:
            n = len(df)
            n_valid = int(round(n * cfg.split.valid_fraction))
            n_valid = max(n_valid, cfg.split.min_valid_size)
            n_valid = min(n_valid, n - 1)

            df['_row_in_series'] = np.arange(n, dtype=int)
            df['_len_series'] = n
            df['_is_valid_raw'] = df['_row_in_series'] >= (n - n_valid)

            min_len = n

        max_lag = int(max(lags))
        max_roll = int(max([*rolling_mean_windows, *rolling_std_windows], default=0))
        max_history = max(max_lag, max_roll)

        if min_len <= h:
            raise ValidationError('Данных недостаточно: min_len <= horizon (в одной из серий)')
        if max_history >= (min_len - h):
            raise ValidationError('Слишком большая глубина истории (lags/rolling) для min_len и horizon')

        # строит таргет y(t+h)
        if sid_col:
            df['_y'] = df.groupby(sid_col)[tgt_col].shift(-h)
            # признак "валидационный пример" зависит от того, где лежит y(t+h)
            df['_is_valid_example'] = (
                df.groupby(sid_col)['_is_valid_raw'].shift(-h).fillna(False).astype(bool)
            )
        else:
            df['_y'] = df[tgt_col].shift(-h)
            df['_is_valid_example'] = df['_is_valid_raw'].shift(-h).fillna(False).astype(bool)

        # признаки: лаги таргета + экзогенные + простые календарные + series_id
        feature_cols: list[str] = []

        # лаги таргета
        for lag in lags:
            col = f'lag_{lag}'
            if sid_col:
                df[col] = df.groupby(sid_col)[tgt_col].shift(lag)
            else:
                df[col] = df[tgt_col].shift(lag)
            feature_cols.append(col)

        
        # скользящие признаки по таргету (строго по прошлому, поэтому shift(1))
        for win in rolling_mean_windows:
            col = f'rolling_mean_{win}'
            if sid_col:
                df[col] = df.groupby(sid_col)[tgt_col].transform(
                    lambda s: s.shift(1).rolling(window=win, min_periods=win).mean()
                )
            else:
                df[col] = df[tgt_col].shift(1).rolling(window=win, min_periods=win).mean()
            feature_cols.append(col)

        for win in rolling_std_windows:
            col = f'rolling_std_{win}'
            if sid_col:
                df[col] = df.groupby(sid_col)[tgt_col].transform(
                    lambda s: s.shift(1).rolling(window=win, min_periods=win).std(ddof=0)
                )
            else:
                df[col] = df[tgt_col].shift(1).rolling(window=win, min_periods=win).std(ddof=0)
            feature_cols.append(col)

        # экзогены
        for col in schema.exogenous_cols:
            feature_cols.append(col)

        # календарные признаки (минимально полезные и дешёвые)
        # (timestamp уже должен быть datetime в parser)
        df['dow'] = df[ts_col].dt.dayofweek.astype(np.int16)
        df['month'] = df[ts_col].dt.month.astype(np.int16)
        feature_cols += ['dow', 'month']

        # series_id как числовой код (если много рядов)
        if sid_col:
            df['series_code'] = pd.factorize(df[sid_col], sort=True)[0].astype(np.int32)
            feature_cols.append('series_code')

        # чистит строки, где нельзя строить supervised пример
        needed = feature_cols + ['_y']
        df2 = df.dropna(subset=needed).copy()

        if df2.empty:
            raise ValidationError('После построения лагов/цели не осталось строк (слишком большие lags/horizon)')

        train_mask = ~df2['_is_valid_example']
        valid_mask = df2['_is_valid_example']

        if not train_mask.any():
            raise ValidationError('Train пустой')
        if not valid_mask.any():
            raise ValidationError('Valid пустой')

        X_train = df2.loc[train_mask, feature_cols]
        y_train = df2.loc[train_mask, '_y']

        X_valid = df2.loc[valid_mask, feature_cols]
        y_valid = df2.loc[valid_mask, '_y']

        return X_train, y_train, X_valid, y_valid, feature_cols