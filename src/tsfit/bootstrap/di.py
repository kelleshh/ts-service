from dishka import Provider, Scope, provide, make_async_container
from dishka.integrations.fastapi import FastapiProvider

from tsfit.application.ports import (
    TrainingRunRepository,
    DatasetParser,
    TimeSeriesDatasetBuilder,
    ModelTrainer,
)
from tsfit.application.usecases import TrainModelUseCase, GetRunUseCase

from tsfit.infrastructure.store.repository_inmemory import InMemoryTrainingRunRepository
from tsfit.infrastructure.dataset.builder_polars import PolarsTimeSeriesDatasetBuilder
from tsfit.infrastructure.dataset.parser_polars import PolarsDatasetParser
from tsfit.infrastructure.training.trainer_xgb import XGBModelTrainer


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def repo(self) -> TrainingRunRepository:
        return InMemoryTrainingRunRepository()

    @provide(scope=Scope.APP)
    def parser(self) -> DatasetParser:
        return PolarsDatasetParser()

    @provide(scope=Scope.APP)
    def builder(self) -> TimeSeriesDatasetBuilder:
        return PolarsTimeSeriesDatasetBuilder()

    @provide(scope=Scope.APP)
    def trainer(self) -> ModelTrainer:
        return XGBModelTrainer()

    @provide(scope=Scope.REQUEST)
    def train_uc(
        self,
        repo: TrainingRunRepository,
        parser: DatasetParser,
        builder: TimeSeriesDatasetBuilder,
        trainer: ModelTrainer,
    ) -> TrainModelUseCase:
        return TrainModelUseCase(repo=repo, parser=parser, builder=builder, trainer=trainer)

    @provide(scope=Scope.REQUEST)
    def get_run_uc(self, repo: TrainingRunRepository) -> GetRunUseCase:
        return GetRunUseCase(repo=repo)


container = make_async_container(AppProvider(), FastapiProvider())