from __future__ import annotations

import os

try:
    from dishka import Provider, Scope, make_container, provide
    from dishka.integrations.fastapi import FastapiProvider
except Exception as e:
    raise RuntimeError('Dishka не установлена') from e

from tsfit.application.ports import (
    DatasetParser,
    FeatureBuilder,
    ModelRegistry,
    ModelTrainer,
    ModelTuner,
    PayloadHasher,
    Predictor,
    Preprocessor,
    SeriesProfiler,
    TargetInspector,
    WalkForwardSplitter,
)
from tsfit.application.usecases import PredictUseCase, TrainModelAutoUseCase, TrainModelUseCase
from tsfit.domain.politics import PlanBuilder

from tsfit.infrastructure.step1_parse.parser_pandas import PandasDatasetParser
from tsfit.infrastructure.step2_preprocess.preprocessor import DefaultPreprocessor
from tsfit.infrastructure.step3_profiling.profiler import DefaultSeriesProfiler
from tsfit.infrastructure.step4_feature_generation.feature_builder import PandasFeatureBuilder
from tsfit.infrastructure.step5_cross_validation.walk_forward import DefaultWalkForwardSplitter
from tsfit.infrastructure.step6_optuna.tuner import OptunaTuner
from tsfit.infrastructure.step7_training.xgb_trainer import XGBoostTrainer
from tsfit.infrastructure.step9_registry.memory_registry import InMemoryModelRegistry
from tsfit.infrastructure.step9_registry.mlflow_registry import MlflowModelRegistry
from tsfit.infrastructure.predict.predictor_xgb import XGBoostPredictor
from tsfit.infrastructure.util.payload_hasher import Sha256PayloadHasher


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_plan_builder(self) -> PlanBuilder:
        return PlanBuilder()

    @provide(scope=Scope.APP)
    def provide_parser(self) -> DatasetParser:
        return PandasDatasetParser()

    @provide(scope=Scope.APP)
    def provide_preprocessor(self) -> Preprocessor:
        return DefaultPreprocessor()

    @provide(scope=Scope.APP)
    def provide_profiler(self) -> SeriesProfiler:
        return DefaultSeriesProfiler()

    @provide(scope=Scope.APP)
    def provide_target_inspector(self) -> TargetInspector:
        return DefaultSeriesProfiler()

    @provide(scope=Scope.APP)
    def provide_feature_builder(self) -> FeatureBuilder:
        return PandasFeatureBuilder()

    @provide(scope=Scope.APP)
    def provide_splitter(self) -> WalkForwardSplitter:
        return DefaultWalkForwardSplitter()

    @provide(scope=Scope.APP)
    def provide_trainer(self) -> ModelTrainer:
        return XGBoostTrainer()

    @provide(scope=Scope.APP)
    def provide_tuner(self) -> ModelTuner:
        return OptunaTuner()

    @provide(scope=Scope.APP)
    def provide_registry(self) -> ModelRegistry:
        backend = os.getenv('TSFIT_REGISTRY', 'mlflow').lower().strip()
        if backend == 'memory':
            return InMemoryModelRegistry()
        # default: mlflow
        return MlflowModelRegistry(experiment_name=os.getenv('TSFIT_MLFLOW_EXPERIMENT', 'tsfit'))

    @provide(scope=Scope.APP)
    def provide_predictor(self) -> Predictor:
        return XGBoostPredictor()

    @provide(scope=Scope.APP)
    def provide_hasher(self) -> PayloadHasher:
        return Sha256PayloadHasher()

    @provide(scope=Scope.REQUEST)
    def provide_train_uc(
        self,
        parser: DatasetParser,
        preprocessor: Preprocessor,
        profiler: SeriesProfiler,
        target_inspector: TargetInspector,
        feature_builder: FeatureBuilder,
        splitter: WalkForwardSplitter,
        trainer: ModelTrainer,
        registry: ModelRegistry,
        hasher: PayloadHasher,
        plan_builder: PlanBuilder,
    ) -> TrainModelUseCase:
        return TrainModelUseCase(
            parser=parser,
            preprocessor=preprocessor,
            profiler=profiler,
            target_inspector=target_inspector,
            feature_builder=feature_builder,
            splitter=splitter,
            trainer=trainer,
            registry=registry,
            hasher=hasher,
            plan_builder=plan_builder,
        )

    @provide(scope=Scope.REQUEST)
    def provide_train_auto_uc(
        self,
        parser: DatasetParser,
        preprocessor: Preprocessor,
        profiler: SeriesProfiler,
        target_inspector: TargetInspector,
        feature_builder: FeatureBuilder,
        tuner: ModelTuner,
        registry: ModelRegistry,
        hasher: PayloadHasher,
        plan_builder: PlanBuilder,
    ) -> TrainModelAutoUseCase:
        return TrainModelAutoUseCase(
            parser=parser,
            preprocessor=preprocessor,
            profiler=profiler,
            target_inspector=target_inspector,
            feature_builder=feature_builder,
            tuner=tuner,
            registry=registry,
            hasher=hasher,
            plan_builder=plan_builder,
        )

    @provide(scope=Scope.REQUEST)
    def provide_predict_uc(
        self,
        parser: DatasetParser,
        preprocessor: Preprocessor,
        feature_builder: FeatureBuilder,
        registry: ModelRegistry,
        predictor: Predictor,
    ) -> PredictUseCase:
        return PredictUseCase(
            parser=parser,
            preprocessor=preprocessor,
            feature_builder=feature_builder,
            registry=registry,
            predictor=predictor,
        )


container = make_container(AppProvider(), FastapiProvider())
