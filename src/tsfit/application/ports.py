from typing import Protocol, Any

from tsfit.domain.training_run import TrainingRun
from tsfit.domain.spec import DatasetSchema, TimeSeriesConfig

Frame = Any # application не должен знать про pandas/polars и pl.DataFrame/pd.DataFrame
Series = Any # аналогичн для series

class TrainingRunRepository(Protocol):
    def save(self, run: TrainingRun) -> None: ...
    def get(self, run_id: str) -> TrainingRun | None: ...
    def find_by_idempotency(self, key: str) -> TrainingRun | None: ...


class DatasetParser(Protocol):
    def parse(self, rows: list[dict[str, Any]], schema: DatasetSchema) -> Frame:
        '''парсит JSON ы табличный формат и делает строгие проверки временнОй оси'''
        ...


class SupervisedDatasetBuilder(Protocol):
    def build_train_valid(
        self,
        frame: Frame,
        schema: DatasetSchema,
        cfg: TimeSeriesConfig,
    ) -> tuple[Frame, Series, Frame, Series, list[str]]:
        '''
        Строит supervised-датасет:
        - признаки (лаги и др.)
        - цель на horizon: y(t+h)
        - разрез train/valid по времени
        Возвращает сплит X_train, y_train, X_valid, y_valid, feature_names
        '''
        ...


class ModelTrainer(Protocol):
    def train_and_eval(
        self,
        X_train: Frame,
        y_train: Series,
        X_valid: Frame,
        y_valid: Series,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        '''Обучает модель и возвращает результат (метрики и тд)'''
        ...