from pydantic import BaseModel, Field
from typing import Any

from tsfit.domain.spec import DatasetSchema, FeatureSpec, TimeSeriesConfig, SplitConfig
from tsfit.domain.training_run import RunStatus


class DatasetSchemaDTO(BaseModel):
    timestamp_col: str
    target_col: str
    series_id_col: str | None = None
    exogenous_cols: list[str] = Field(default_factory=list)

    def to_domain(self) -> DatasetSchema:
        return DatasetSchema(
            timestamp_col=self.timestamp_col,
            target_col=self.target_col,
            series_id_col=self.series_id_col,
            exogenous_cols=self.exogenous_cols,
        )
    
class FeatureSpecDTO(BaseModel):
    lags: list[int] = Field(min_length=1)
    rolling_mean_windows: list[int] = Field(default_factory=list)
    rolling_std_windows: list[int] = Field(default_factory=list)
    
    def to_domain(self) -> FeatureSpec:
        return FeatureSpec(
            lags=self.lags,
            rolling_mean_windows=self.rolling_mean_windows,
            rolling_std_windows=self.rolling_std_windows,)
    

class SplitConfigDTO(BaseModel):
    valid_fraction: float = 0.2
    min_valid_size: int = 5
    
    def to_domain(self) -> SplitConfig:
        return SplitConfig(
            valid_fraction=self.valid_fraction,
            min_valid_size=self.min_valid_size,
        )


class TimeSeriesConfigDTO(BaseModel):
    horizon: int
    features: FeatureSpecDTO
    split: SplitConfigDTO = Field(default_factory=SplitConfigDTO)

    def to_domain(self) -> TimeSeriesConfig:
        return TimeSeriesConfig(
            horizon=self.horizon,
            features=self.features.to_domain(),
            split=self.split.to_domain(),
        )


class FitRequest(BaseModel):
    dataset: list[dict[str, Any]]
    dataset_schema: DatasetSchemaDTO
    ts: TimeSeriesConfigDTO
    params: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class FitResponse(BaseModel):
    run_id: str
    status: RunStatus
    created: bool
    metrics: dict[str, float] | None = None
