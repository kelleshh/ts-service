from __future__ import annotations

from dataclasses import asdict

from tsfit.domain.value_objects.meta import ModelMeta
from tsfit.application.ports import (
    DatasetParser,
    FeatureBuilder,
    ModelRegistry,
    ModelTrainer,
    PayloadHasher,
    Preprocessor,
    SeriesProfiler,
    TargetInspector,
    WalkForwardSplitter,
)
from tsfit.application.usecases.commands import FitCommand
from tsfit.application.usecases.results import FitResult
from tsfit.domain.exceptions import ConflictError, ValidationError
from tsfit.domain.politics import PlanBuilder
from tsfit.domain.value_objects import DatasetSchema, TrainingConfig

# Use Case (Application Service)
class TrainModelUseCase:
    def __init__(
        self,
        *,
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
    ) -> None:
        self._parser = parser
        self._preprocessor = preprocessor
        self._profiler = profiler
        self._target_inspector = target_inspector
        self._feature_builder = feature_builder
        self._splitter = splitter
        self._trainer = trainer
        self._registry = registry
        self._hasher = hasher
        self._plan_builder = plan_builder

    def execute(self, cmd: FitCommand) -> FitResult:
        if not cmd.idempotency_key:
            raise ValidationError('idempotency_key обязателен')
        if cmd.horizon < 1:
            raise ValidationError('horizon должен быть >= 1')
        
        
        
        # парсим schema и training из cmd через .make()
        schema = DatasetSchema.make(
            cmd.timestamp_col, 
            cmd.target_col, 
            cmd.exog_cols
            )
        training = TrainingConfig.make(
            cmd.primary_metric,
            cmd.metrics,
            cmd.model_params,
            cmd.early_stopping_rounds,
            cmd.n_estimators_cap,
        )

        # хэш
        payload_hash = self._hasher.hash_fit_payload(
            rows=cmd.rows,
            schema=schema,
            horizon=cmd.horizon,
            training=training,
            tuning=None,
            policy_version=self._plan_builder.policy_version,
        )

        # идемпотентность
        found = self._registry.find_by_idempotency_key(cmd.idempotency_key)
        if found is not None:
            model_id, old_hash = found
            if old_hash != payload_hash:
                raise ConflictError('idempotency_key уже использован с другим payload')
            summary = self._registry.load_summary(model_id)
            return FitResult(model_id=model_id, created=False, summary=summary)
        
        # парсинг датасета
        raw = self._parser.parse(cmd.rows)
        # строим план препроцессинга
        preprocess_plan = self._plan_builder.build_preprocess_plan()        
        # препроцессим
        cleaned = self._preprocessor.preprocess(raw, schema, preprocess_plan)

        # проверка метрики mape
        if training.primary_metric == 'mape':
            frac = self._target_inspector.frac_abs_y_lt_eps(
                cleaned,
                schema,
                eps=training.mape_eps,
            )
            if frac > training.mape_zero_frac_threshold:
                raise ValidationError(
                    f"mape запрещен: доля |y| < eps = {frac:.6f} > {training.mape_zero_frac_threshold} "
                    f"(eps={training.mape_eps}) используйте smape"
                )
            
        # профилирование датасета
        profile = self._profiler.profile(cleaned, schema, min_train_ratio=0.15)
        # строим план (препроцессинг не нужен здесь, уже отпрепроцессили)
        _, feature_plan, cv_plan = self._plan_builder.build_all(profile, cmd.horizon)

        # строим supervised-датасет
        supervised = self._feature_builder.build_supervised(
            cleaned,
            schema,
            feature_plan,
            horizon=cmd.horizon,
        )

        # сплиттим по схеме и плану кросс валидации на фолды и x y
        full_sd = self._feature_builder.split_xy(supervised, schema)
        folds = self._splitter.split(supervised, cv_plan)
        # сплит по фолдам на x и y
        fold_xy: list[tuple] = []
        for fold in folds:
            train_sd = self._feature_builder.split_xy(fold.train, schema)
            valid_sd = self._feature_builder.split_xy(fold.valid, schema)
            fold_xy.append((train_sd, valid_sd))

        # трейним с кросс валидацией
        report = self._trainer.train_with_walk_forward(
            full=full_sd,
            folds=fold_xy,
            feature_names=full_sd.feature_names,
            config=training,
        )

        # саммари
        summary = _build_summary(report)
        # метаданные обучения
        meta = ModelMeta(
            schema=schema,
            preprocess_plan=preprocess_plan,
            feature_plan=feature_plan,
            cv_plan=cv_plan,
            horizon=cmd.horizon,
            policy_version=self._plan_builder.policy_version,
            training_config=training,
            feature_names=report.feature_names,
        )

        # логируем модель и метаданные ее
        model_id = self._registry.save(
            idempotency_key=cmd.idempotency_key,
            payload_hash=payload_hash,
            report=report,
            meta=meta,
            summary=summary,
            extra={
                'profile': asdict(profile),
            },
        )

        return FitResult(model_id=model_id, created=True, summary=summary)


def _build_summary(report) -> dict:
    train_metrics = {
        k: {'mean': float(v.mean), 'std': float(v.std)}
        for k, v in report.metrics.train.items()
    }
    valid_metrics = {
        k: {'mean': float(v.mean), 'std': float(v.std)}
        for k, v in report.metrics.valid.items()
    }

    fi_items = sorted(report.feature_importance_gain.items(), key=lambda kv: kv[1], reverse=True)
    feature_importance = [{'name': k, 'importance': float(v)} for k, v in fi_items]

    return {
        'model_params': dict(report.model_params),
        'training_params': dict(report.training_params),
        'metrics': {
            'train': train_metrics,
            'valid': valid_metrics,
        },
        'feature_importance': feature_importance,
    }
