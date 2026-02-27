from __future__ import annotations

from tsfit.application.ports import DatasetParser, FeatureBuilder, ModelRegistry, Predictor, Preprocessor
from tsfit.application.usecases.commands import PredictCommand
from tsfit.application.usecases.results import PredictResult
from tsfit.domain.exceptions import ValidationError


# Use Case (Application Service)
class PredictUseCase:
    def __init__(
        self,
        *,
        parser: DatasetParser,
        preprocessor: Preprocessor,
        feature_builder: FeatureBuilder,
        registry: ModelRegistry,
        predictor: Predictor,
    ) -> None:
        self._parser = parser
        self._preprocessor = preprocessor
        self._feature_builder = feature_builder
        self._registry = registry
        self._predictor = predictor

    def execute(self, cmd: PredictCommand) -> PredictResult:
        if not cmd.model_id:
            raise ValidationError('model_id обязателен')

        model, meta = self._registry.load_model_and_meta(cmd.model_id)

        raw = self._parser.parse(cmd.rows)
        cleaned = self._preprocessor.preprocess(raw, meta.schema, meta.preprocess_plan)
        x_last, feature_names = self._feature_builder.build_last_x(cleaned, meta.schema, meta.feature_plan)

        if tuple(feature_names) != tuple(meta.feature_names):
            raise ValidationError('feature_names не совпали с обучением')

        pred = self._predictor.predict_one(model, x_last)
        return PredictResult(prediction=float(pred))
