from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any

from tsfit.domain.value_objects import (
    DatasetSchemaValueObject, 
    FeatureSpecValueObject, 
    TimeSeriesConfigValueObject, 
    SplitConfigValueObject,
    TrainingConfigValueObject,
    TuningConfigValueObject,
)
from tsfit.domain.entities import RunStatus

ALLOWED_EVAL_METRICS: frozenset[str] = frozenset(
    {
        'rmse',
        'rmsle',
        'mae',
        'mape',
        'mphe',
        'logloss',
        'error',
        'merror',
        'mlogloss',
        'auc',
        'aucpr',
        'pre',
        'ndcg',
        'map',
        'poisson-nloglik',
        'gamma-nloglik',
        'cox-nloglik',
        'gamma-deviance',
        'tweedie-nloglik',
        'aft-nloglik',
        'interval-regression-accuracy',
    }
)

# DTO (обмен данными для API)

class DatasetSchemaDTO(BaseModel):
    '''
    Схема датасета: какие ключи что означают
    '''

    model_config = ConfigDict(extra='forbid')

    timestamp_col: str
    target_col: str
    series_id_col: str | None = None
    exogenous_cols: list[str] = Field(default_factory=list)

    def to_domain(self) -> DatasetSchemaValueObject:
        return DatasetSchemaValueObject(
            timestamp_col=self.timestamp_col,
            target_col=self.target_col,
            series_id_col=self.series_id_col,
            exogenous_cols=self.exogenous_cols,
        )


class FeatureSpecDTO(BaseModel):
    '''
    Настройки генерации признаков (лаги, скользящие статистики и т.д.)
    '''

    model_config = ConfigDict(extra='forbid')

    lags: list[int] = Field(min_length=1)
    rolling_mean_windows: list[int] = Field(default_factory=list)
    rolling_std_windows: list[int] = Field(default_factory=list)
    rolling_min_windows: list[int] = Field(default_factory=list)
    rolling_max_windows: list[int] = Field(default_factory=list)
    diff_lags: list[int] = Field(default_factory=list)

    def to_domain(self) -> FeatureSpecValueObject:
        return FeatureSpecValueObject(
            lags=self.lags,
            rolling_mean_windows=self.rolling_mean_windows,
            rolling_std_windows=self.rolling_std_windows,
            rolling_min_windows=self.rolling_min_windows,
            rolling_max_windows=self.rolling_max_windows,
            diff_lags=self.diff_lags,
        )


class SplitConfigDTO(BaseModel):
    '''
    Настройки сплита (train/valid)
    '''

    model_config = ConfigDict(extra='forbid')

    valid_fraction: float = 0.2
    min_valid_size: int = 5

    def to_domain(self) -> SplitConfigValueObject:
        return SplitConfigValueObject(
            valid_fraction=self.valid_fraction,
            min_valid_size=self.min_valid_size,
        )


class TimeSeriesConfigDTO(BaseModel):
    '''
    Конфиг построения временного ряда
    '''

    model_config = ConfigDict(extra='forbid')

    horizon: int
    features: FeatureSpecDTO
    split: SplitConfigDTO = Field(default_factory=SplitConfigDTO)

    def to_domain(self) -> TimeSeriesConfigValueObject:
        return TimeSeriesConfigValueObject(
            horizon=self.horizon,
            features=self.features.to_domain(),
            split=self.split.to_domain(),
        )


class TuningConfigDTO(BaseModel):
    '''
    Конфиг автоподбора гиперпараметров
    '''

    model_config = ConfigDict(extra='forbid')

    n_trials: int = 20
    timeout_sec: int | None = 60

    def to_domain(self) -> TuningConfigValueObject:
        return TuningConfigValueObject(
            n_trials=self.n_trials,
            timeout_sec=self.timeout_sec,
        )


class ModelParamsDTO(BaseModel):
    '''
    Явно описанные параметры обучения, которые мы разрешаем принимать от пользователя
    '''
    model_config = ConfigDict(extra='forbid')

    # базовые
    objective: str | None = None
    n_estimators: int | None = None
    learning_rate: float | None = None
    max_depth: int | None = None

    # выбор подвыборок
    subsample: float | None = None
    colsample_bytree: float | None = None

    # регуляризация
    min_child_weight: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None

    # параллелизм
    n_jobs: int | None = None

    # воспроизводимость
    random_state: int | None = None

    def to_model_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.model_dump().items():
            if v is not None:
                out[k] = v
        return out


class TrainingConfigDTO(BaseModel):
    '''
    Конфиг обучения по фиксированным параметрам без тюнинга.

    - запрещаются неизвестные параметры через modelParamsDTO
    - проверка корректности имен метрик по инвариантам
    '''

    model_config = ConfigDict(extra='forbid')

    model_params: ModelParamsDTO = Field(default_factory=ModelParamsDTO)
    metrics: list[str] = Field(default_factory=lambda: ['rmse', 'mae'])
    primary_metric: str = 'rmse'

    def to_domain(self) -> TrainingConfigValueObject:
        return TrainingConfigValueObject(
            model_params=self.model_params.to_model_params(),
            metrics=self.metrics,
            primary_metric=self.primary_metric,
        )
    
    @field_validator('metrics')
    @classmethod
    def _validate_metrics(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list) or not v:
            raise ValueError('training.metrics должен быть непустым списком')

        cleaned = [m.strip() for m in v]
        if any(not m for m in cleaned):
            raise ValueError('training.metrics не должен содержать пустые строки')
        if len(set(cleaned)) != len(cleaned):
            raise ValueError('training.metrics не должен содержать дубликаты')

        unknown = [m for m in cleaned if m not in ALLOWED_EVAL_METRICS]
        if unknown:
            raise ValueError(
                'training.metrics содержит неизвестные метрики: '
                f'{unknown}. Допустимо: {sorted(ALLOWED_EVAL_METRICS)}'
            )

        return cleaned

    @field_validator('primary_metric')
    @classmethod
    def _validate_primary_metric(cls, v: str) -> str:
        pm = v.strip() if isinstance(v, str) else ''
        if not pm:
            raise ValueError('training.primary_metric должен быть непустой строкой')
        if pm not in ALLOWED_EVAL_METRICS:
            raise ValueError(
                'training.primary_metric содержит неизвестную метрику. '
                f'Допустимо: {sorted(ALLOWED_EVAL_METRICS)}'
            )
        return pm
    

class DataProfileDTO(BaseModel):
    '''
    Профиль данных после построения признаков

    Фиксированная часть результата: по этим числам можно 
    понять, насколько много данных реально дошло до обучения
    '''

    model_config = ConfigDict(extra='forbid')

    n_rows_raw: int
    n_total_after_features: int
    n_train: int
    n_valid: int
    n_features: int


class Stage1CandidateDTO(BaseModel):
    model_config = ConfigDict(extra='forbid')

    learning_rate: float
    best_iteration: int
    best_value: float


class Stage1ReportDTO(BaseModel):
    model_config = ConfigDict(extra='forbid')

    learning_rates: list[float]
    candidates: list[Stage1CandidateDTO]
    chosen_learning_rate: float
    chosen_n_estimators: int
    chosen_best_value: float


class Stage2ReportDTO(BaseModel):
    model_config = ConfigDict(extra='forbid')

    direction: str
    n_trials_requested: int
    n_trials_used: int
    n_trials_ran: int
    best_value: float
    best_params: dict[str, Any]


class TuningReportDTO(BaseModel):
    '''
    Отчет о подборе параметров
    '''

    model_config = ConfigDict(extra='allow') # на будущее если отчет расширять

    strategy_used: str
    data_bucket: str
    data_profile: DataProfileDTO
    stage1: Stage1ReportDTO
    stage2: Stage2ReportDTO

# СХЕМЫ ВХОДА И ВЫХОДА


class FitRequest(BaseModel):
    '''
    Тело запроса POST /fit
    '''

    model_config = ConfigDict(extra='forbid')

    dataset: list[dict[str, Any]]
    dataset_schema: DatasetSchemaDTO
    ts: TimeSeriesConfigDTO
    training: TrainingConfigDTO = Field(default_factory=TrainingConfigDTO)
    idempotency_key: str | None = None


class FitResponse(BaseModel):
    '''
    Ответ POST /fit
    '''

    run_id: str
    status: RunStatus
    created: bool
    metrics: dict[str, float] | None = None
    data_profile: DataProfileDTO | None = None


class FitAutoRequest(BaseModel):
    '''
    Тело запроса POST /fit/auto
    '''

    model_config = ConfigDict(extra='forbid')

    dataset: list[dict[str, Any]]
    dataset_schema: DatasetSchemaDTO
    ts: TimeSeriesConfigDTO
    training: TrainingConfigDTO = Field(default_factory=TrainingConfigDTO)
    tuning: TuningConfigDTO = Field(default_factory=TuningConfigDTO)
    idempotency_key: str | None = None


class FitAutoResponse(BaseModel):
    '''
    Ответ POST /fit/auto
    '''

    run_id: str
    status: RunStatus
    created: bool
    metrics: dict[str, float] | None = None
    data_profile: DataProfileDTO | None = None
    
    # краткий отчет о подборе
    tuning_report: TuningReportDTO | None = None
