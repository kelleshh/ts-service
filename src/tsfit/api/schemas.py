from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
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

# DTOшки

class DatasetSchemaDTO(BaseModel):
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
            diff_lags=self.diff_lags,)
    

class SplitConfigDTO(BaseModel):
    model_config = ConfigDict(extra='forbid')
    valid_fraction: float = 0.2
    min_valid_size: int = 5
    
    def to_domain(self) -> SplitConfigValueObject:
        return SplitConfigValueObject(
            valid_fraction=self.valid_fraction,
            min_valid_size=self.min_valid_size,
        )


class TimeSeriesConfigDTO(BaseModel):
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
    model_config = ConfigDict(extra='forbid')
    
    n_trials: int = 20
    timeout_sec: int | None = 60

    def to_domain(self) -> TuningConfigValueObject:
        return TuningConfigValueObject(
            n_trials=self.n_trials,
            timeout_sec=self.timeout_sec,
        )
    

class XGBParamsDTO(BaseModel):
    '''
    Явно описанные параметры обучения, которые мы разрешаем принимать от пользователя
    '''
    model_config = ConfigDict(extra='forbid')

    # базовые
    objective: str | None = None
    n_estimators: int | None = None
    learning_rate: float | None = None
    max_depth: int | None = None

    # сэмплинг
    subsample: float | None = None
    colsample_bytree: float | None = None

    # регуляризация
    min_child_weight: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None

    # воспроизводимость
    random_state: int | None = None

    def to_xgb_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.model_dump().items():
            if v is not None:
                out[k] = v
        return out


class TrainingConfigDTO(BaseModel):
    '''
    Конфиг обучения по фиксированным параметрам без тюнинга
    '''
    model_config = ConfigDict(extra='forbid')
    xgb_params: dict[str, Any] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=lambda: ['rmse', 'mae']) # TODO: сделать поддержку всех метрик, доступных в xgboost
    primary_metric: str = 'rmse'

    def to_domain(self) -> TrainingConfigValueObject:
        return TrainingConfigValueObject(
            xgb_params=self.xgb_params,
            metrics=self.metrics,
            primary_metric=self.primary_metric,
        )


# СХЕМЫ ВХОДА И ВЫХОДА

class FitRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dataset: list[dict[str, Any]]
    dataset_schema: DatasetSchemaDTO
    ts: TimeSeriesConfigDTO
    training: TrainingConfigDTO = Field(default_factory=TrainingConfigDTO)
    idempotency_key: str | None = None


class FitResponse(BaseModel):
    run_id: str
    status: RunStatus
    created: bool
    metrics: dict[str, float] | None = None


class FitAutoRequest(BaseModel):
    '''
    Обучение с автоподбором гиперпараметров.
    '''
    model_config = ConfigDict(extra='forbid')

    dataset: list[dict[str, Any]]
    dataset_schema: DatasetSchemaDTO
    ts: TimeSeriesConfigDTO
    training: TrainingConfigDTO = Field(default_factory=TrainingConfigDTO)
    tuning: TuningConfigDTO = Field(default_factory=TuningConfigDTO)
    idempotency_key: str | None = None


class FitAutoResponse(BaseModel):
    run_id: str
    status: RunStatus
    created: bool
    metrics: dict[str, float] | None = None

    # краткий отчет о подборе
    tuning: dict[str, Any] | None = None
