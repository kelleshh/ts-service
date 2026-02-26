from typing import Any
from pydantic import Field
from presentation.apimodel import APIModel


# DTO-шки


class DatasetSchemaDTO(APIModel):
    timestamp_col: str = Field(min_length=1)
    target_col: str = Field(min_length=1)
    exog_cols: tuple[str, ...] = Field(default_factory=tuple)


class TrainingDTO(APIModel):
    primary_metric: str = Field(default='smape', min_length=1)
    metrics: tuple[str, ...] = Field(default_factory=lambda: ('smape', 'mae', 'rmse'))
    model_params: dict[str, Any] = Field(default_factory=dict)

    early_stopping_rounds: int = Field(default=50, ge=0)
    n_estimators_cap: int = Field(default=2000, ge=1)


class TuningDTO(APIModel):
    n_trials: int = Field(default=50, ge=1)
    timeout_sec: int | None = Field(default=None, ge=1)


# родительский класс реквестов


class FitBaseRequest(APIModel):
    idempotency_key: str = Field(min_length=1)
    dataset: list[dict[str, Any]] = Field(min_length=1)
    dataset_schema: DatasetSchemaDTO
    horizon: int = Field(default=1, ge=1)
    training: TrainingDTO = Field(default_factory=TrainingDTO)


# request, auto request


class FitRequest(FitBaseRequest):
    pass

class FitAutoRequest(FitBaseRequest):
    tuning: TuningDTO = Field(default_factory=TuningDTO)


# response, auto response


class FitResponse(APIModel):
    model_id: str = Field(min_length=1)
    created: bool
    summary: dict[str, Any] = Field(default_factory=dict)


class FitAutoResponse(FitResponse):
    pass


