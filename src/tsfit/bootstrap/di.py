from dishka import Provider, Scope, provide, make_async_container
from dishka.integrations.fastapi import FastapiProvider

from tsfit.application.ports import (
    TrainingRunRepository,
    DatasetParser,
    SupervisedDatasetBuilder,
    ModelTrainer,
)
from tsfit.application.usecases import TrainModelUseCase, GetRunUseCase

from tsfit.infrastructure.store_inmemory import InMemoryTrainingRunRepository
from tsfit.infrastructure.feature_builder_pandas import PandasSupervisedDatasetBuilder
from tsfit.infrastructure.dataset_parser_pandas import PandasDatasetParser
from tsfit.infrastructure.trainer_xgb import XGBModelTrainer


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def repo(self) -> TrainingRunRepository:
        return InMemoryTrainingRunRepository()

    @provide(scope=Scope.APP)
    def parser(self) -> DatasetParser:
        return PandasDatasetParser()

    @provide(scope=Scope.APP)
    def builder(self) -> SupervisedDatasetBuilder:
        return PandasSupervisedDatasetBuilder()

    @provide(scope=Scope.APP)
    def trainer(self) -> ModelTrainer:
        return XGBModelTrainer()

    @provide(scope=Scope.REQUEST)
    def train_uc(
        self,
        repo: TrainingRunRepository,
        parser: DatasetParser,
        builder: SupervisedDatasetBuilder,
        trainer: ModelTrainer,
    ) -> TrainModelUseCase:
        return TrainModelUseCase(repo=repo, parser=parser, builder=builder, trainer=trainer)

    @provide(scope=Scope.REQUEST)
    def get_run_uc(self, repo: TrainingRunRepository) -> GetRunUseCase:
        return GetRunUseCase(repo=repo)


container = make_async_container(AppProvider(), FastapiProvider())