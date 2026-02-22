from dataclasses import dataclass, field
from tsfit.domain.errors import ValidationError

@dataclass(frozen=True)
class DatasetSchema:
    '''
    Схема колонок в датасете
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
class FeatureSpec:
    '''
    Спецификация для генерации признаков
    '''
    lags: list[int]
    # TODO: добавить другие фичи!

    def validate(self) -> None:
        if not self.lags:
            raise ValidationError('lags должен быть не пустым')
        if any((not isinstance(l, int)) for l in self.lags):
            raise ValidationError('lags должен быть списком целых чисел')
        if any(l <= 0 for l in self.lags):
            raise ValidationError('lags должен содержать только числа > 0')
        if len(set(self.lags)) != len(self.lags):
            raise ValidationError('lags не должен содержать дубликаты')


@dataclass(frozen=True)
class SplitConfig:
    '''
    Конфиг сплитинга по времени
    '''
    valid_fraction: float = 0.2
    min_valid_size: int = 5

    def validate(self) -> None:
        if not (0.0 < self.valid_fraction < 1.0):
            raise ValidationError('valid_fraction должен быть между 0 и 1 (не включая границы)')
        if self.min_valid_size <= 0:
            raise ValidationError('min_valid_size должен быть > 0')

@dataclass(frozen=True)
class TimeSeriesConfig:
    '''
    Конфиг постановки временного ряда
    '''
    horizon: int
    features: FeatureSpec
    split: SplitConfig = field(default_factory=SplitConfig)

    def validate(self) -> None:
        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValidationError('horizon должен целым быть > 0')
        self.features.validate() # валидируем фичи
        self.split.validate() # валидируем сплиты
        


'''
в домене должны быть (я не знаю что тут избыточно, что уже реализовано а чего реально не хватает):
1) сущности (entity): туда надо добавить:
- TimeSeries (сам временной ряд) (series_id, entity_id, timezone, native_frequency, granularity: bool, правила допустимых пропусков хз что это значит я переписываю под диктовку)
- TimeGrid (сетка времени) (grid_frequency, start ?, end ?, функция генерации идеальной сетки timestamps)
- Signal (фича источник) (role: target или feature)
- ResamplingSpec (спецификация приведения частоты) (from_frequency -> to_frequency, method (agg, ffill, asof-join, interpolation запрещено разрешено), allowed_aggregations, missiong_policy 9fill, zero, keep nan, flag)
- FeatureSpec (список фичей) Каждая фича знает из какого Signal она и как ресемплится
- DatasetVersion (dataset_id, time_window, grid_frequency, чексумма входа)
'''