from dataclasses import dataclass, field
from typing import Any
from tsfit.domain.exceptions import ValidationError
from tsfit.domain.metrics_invariants import BASE_XGBOOST_EVAL_METRICS

@dataclass(frozen=True)
class DatasetSchemaValueObject:
    '''
    Схема колонок в датасете
    - время
    - целевая переменная (то, что прогнозируем)
    - идентификатор ряда (если рядов много)
    - какие колонки считаются экзогенными признаками
    '''
    timestamp_col: str
    target_col: str
    series_id_col: str | None
    exogenous_cols: list[str]

    def validate(self) -> None:
        def _non_empty(name: str, v: str) -> None:
            if not isinstance(v, str) or not v.strip():
                raise ValidationError(f'{name} должен быть непустой строкой')

        _non_empty('timestamp_col', self.timestamp_col)
        _non_empty('target_col', self.target_col)
        if self.series_id_col is not None:
            _non_empty('series_id_col', self.series_id_col)

        cols = [self.timestamp_col, self.target_col, *self.exogenous_cols]
        if self.series_id_col:
            cols.append(self.series_id_col)

        if len(set(cols)) != len(cols):
            raise ValidationError('dataset_schema: имена колонок не должны пересекаться')

@dataclass(frozen=True)
class FeatureSpecValueObject:
    '''
    Спецификация для генерации признаков
    !!!: скользящие статистики считаются только по прошлым значениям
    '''
    lags: list[int]
    rolling_mean_windows: list[int]
    rolling_std_windows: list[int]
    rolling_min_windows: list[int] = field(default_factory=list)
    rolling_max_windows: list[int] = field(default_factory=list)
    diff_lags: list[int] = field(default_factory=list)

    def validate(self) -> None:
        def _validate_pos_int_list(name: str, values: list[int]) -> None:
            if any((not isinstance(v, int)) for v in values):
                raise ValidationError(f'{name} должен быть списком целых чисел')
            if any(v <= 0 for v in values):
                raise ValidationError(f'{name} должен содержать только числа > 0')
            if len(set(values)) != len(values):
                raise ValidationError(f'{name} не должен содержать дубликаты')
        if not self.lags:
            raise ValidationError('lags должен быть не пустым')
        _validate_pos_int_list('lags', self.lags)
        _validate_pos_int_list('rolling_mean_windows', self.rolling_mean_windows)
        _validate_pos_int_list('rolling_std_windows', self.rolling_std_windows)
        _validate_pos_int_list('rolling_min_windows', self.rolling_min_windows)
        _validate_pos_int_list('rolling_max_windows', self.rolling_max_windows)
        _validate_pos_int_list('diff_lags', self.diff_lags)


@dataclass(frozen=True)
class SplitConfigValueObject:
    '''
    Конфиг сплитинга по времени
    - ранние даты идут в train
    - более поздние даты идут в valid

    рандомное перемешивание запрещено
    '''
    valid_fraction: float = 0.2
    min_valid_size: int = 5

    def validate(self) -> None:
        if not (0.0 < self.valid_fraction < 1.0):
            raise ValidationError('valid_fraction должен быть между 0 и 1 (не включая границы)')
        if self.min_valid_size <= 0:
            raise ValidationError('min_valid_size должен быть > 0')

@dataclass(frozen=True)
class TimeSeriesConfigValueObject:
    '''
    Конфиг постановки временного ряда
    '''
    horizon: int
    features: FeatureSpecValueObject
    split: SplitConfigValueObject = field(default_factory=SplitConfigValueObject)

    def validate(self) -> None:
        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValidationError('horizon должен целым быть > 0')
        self.features.validate() # валидируем фичи
        self.split.validate() # валидируем сплиты
        

@dataclass(frozen=True)
class TuningConfigValueObject:
    '''
    Настройки опционального подбора гиперпараметров
    '''

    n_trials: int = 20
    timeout_sec: int | None = 60

    def validate(self) -> None:
        if not isinstance(self.n_trials, int) or self.n_trials <= 0:
            raise ValidationError('tuning.n_trials должен быть целым числом > 0')
        if self.n_trials > 50:
            raise ValidationError('tuning.n_trials слишком большой (max=50)')

        if self.timeout_sec is not None:
            if not isinstance(self.timeout_sec, int) or self.timeout_sec <= 0:
                raise ValidationError('tuning.timeout_sec должен быть целым числом > 0 или null')
            if self.timeout_sec > 600:
                raise ValidationError('tuning.timeout_sec слишком большой (max=600)')

@dataclass(frozen=True)
class TrainingConfigValueObject:
    '''
    Конфиг обучения модели по фиксированному набору параметров

    список метрик и primary_metric проверяем как доменный инвариант, потому что это часть контракта
    '''

    xgb_params: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=lambda: ['rmse', 'mae'])
    primary_metric: str = 'rmse'

    def validate(self) -> None:
        if not isinstance(self.xgb_params, dict) or any(not isinstance(k, str) for k in self.xgb_params.keys()):
            raise ValidationError('xgb_params должен быть словарём с строковыми ключами')

        allowed = BASE_XGBOOST_EVAL_METRICS
        if not isinstance(self.metrics, list) or not self.metrics:
            raise ValidationError('metrics должен быть непустым списком')
        if any((not isinstance(m, str)) for m in self.metrics):
            raise ValidationError('metrics должен быть списком строк')
        if len(set(self.metrics)) != len(self.metrics):
            raise ValidationError('metrics не должен содержать дубликаты')
        unknown = [m for m in self.metrics if m not in allowed]
        if unknown:
            raise ValidationError(f'metrics содержит неизвестные метрики: {unknown}. Допустимо: {sorted(allowed)}')

        if self.primary_metric not in allowed:
            raise ValidationError(f'primary_metric должен быть одной из {sorted(allowed)}')
        if self.primary_metric not in self.metrics:
            raise ValidationError('primary_metric должен входить в metrics')