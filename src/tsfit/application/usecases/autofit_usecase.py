from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Callable
import uuid

from tsfit.application.ports import (
    TrainingRunRepository,
    DatasetParser,
    TimeSeriesDatasetBuilder,
    ModelTrainer,
    HyperparameterTuner,
    BuiltDataset,
)
from tsfit.domain.exceptions import IdempotencyConflict, TrainingFailed, ValidationError
from tsfit.domain.entities import TrainingRunEntity, RunStatus
from tsfit.domain.value_objects import (
    DatasetSchemaValueObject, 
    TimeSeriesConfigValueObject, 
    TrainingConfigValueObject,
    TuningConfigValueObject,
    )
from tsfit.domain.rules import rule_validate_rows_have_columns
from tsfit.application.utils.payload_hashing import hash_fit_auto_payload



@dataclass(frozen=True)
class TrainModelAutoRequest:
    '''
    Запрос на обучение модели с автоподбором гиперпараметров
    '''
    dataset_rows: list[dict[str, Any]]
    dataset_schema: DatasetSchemaValueObject
    time_series: TimeSeriesConfigValueObject
    training: TrainingConfigValueObject
    tuning: TuningConfigValueObject
    idempotency_key: str | None = None


@dataclass(frozen=True)
class TrainModelAutoResult:
    '''
    Результат обучения с автоподбором
    '''
    run_id: str
    status: RunStatus
    created: bool
    metrics: dict[str, float] | None
    tuning: dict[str, Any] | None


class TrainModelAutoUseCase:
    '''
    Сценарий обучения XGBoost на временных рядах с автоподбором гиперпараметров
    - сначала подбираем параметры (tuner)
    - затем делаем финальное обучение (trainer)
    '''
    def __init__(
        self,
        repo: TrainingRunRepository,
        parser: DatasetParser,
        builder: TimeSeriesDatasetBuilder,
        trainer: ModelTrainer,
        tuner: HyperparameterTuner,
        id_gen: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repo = repo
        self.parser = parser
        self.builder = builder
        self.trainer = trainer
        self.tuner = tuner
        self.id_gen = id_gen or (lambda: str(uuid.uuid4()))
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def execute(self, req: TrainModelAutoRequest) -> TrainModelAutoResult:
        # доменные проверки
        req.dataset_schema.validate()
        req.time_series.validate()
        req.training.validate()
        req.tuning.validate()
        rule_validate_rows_have_columns(req.dataset_rows, req.dataset_schema)

        # хэш
        payload_hash = hash_fit_auto_payload(
            dataset_rows=req.dataset_rows,
            dataset_schema=req.dataset_schema,
            time_series=req.time_series,
            training=req.training,
            tuning=req.tuning,
        )

        # идемпотентность
        if req.idempotency_key:
            existing = self.repo.find_by_idempotency(req.idempotency_key)
            if existing:
                if existing.payload_hash and existing.payload_hash != payload_hash:
                    raise IdempotencyConflict('Одинаковый idempotency_key, но разные данные/параметры')
                metrics = None
                tuning = None
                if existing.result and isinstance(existing.result, dict):
                    if isinstance(existing.result.get('metrics'), dict):
                        metrics = existing.result['metrics']
                    if isinstance(existing.result.get('tuning'), dict):
                        tuning = existing.result['tuning']
                return TrainModelAutoResult(
                    run_id=existing.run_id,
                    status=existing.status,
                    created=False,
                    metrics=metrics,
                    tuning=tuning,
                )

        run = TrainingRunEntity.new_pending(
            run_id=self.id_gen(),
            created_at=self.clock(),
            idempotency_key=req.idempotency_key,
            payload_hash=payload_hash,
        )
        self.repo.save(run)

        try:
            run.mark_running()
            self.repo.save(run)

            frame = self.parser.parse(req.dataset_rows, req.dataset_schema)

            built: BuiltDataset = self.builder.build_train_valid(
                frame=frame,
                schema=req.dataset_schema,
                cfg=req.time_series,
            )

            # дефолтные параметры
            defaults: dict[str, Any] = {
                'objective': 'reg:squarederror',
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': 1,
            }
            user_params = dict(req.training.xgb_params or {})
            base_params = {**defaults, **user_params}

            # подбор гиперпараметров
            best_params, tuning_report = self.tuner.tune(
                dataset=built,
                base_params=base_params,
                primary_metric=req.training.primary_metric,
                tuning=req.tuning,
            )

            # финальное обучение уже по подобранным гиперпеараметрам (без optuna)
            final_params = {**base_params, **best_params}
            final_training = TrainingConfigValueObject(
                xgb_params=final_params,
                metrics=req.training.metrics,
                primary_metric=req.training.primary_metric,
            )
            result = self.trainer.train_and_eval(dataset=built, training=final_training)

            # доп данные о подборе сохраняем в результат
            result.setdefault('feature_names', built.feature_names)
            result['tuning'] = tuning_report

            run.mark_done(result=result)
            self.repo.save(run)

            metrics = result.get('metrics')
            return TrainModelAutoResult(
                run_id=run.run_id,
                status=run.status,
                created=True,
                metrics=metrics if isinstance(metrics, dict) else None,
                tuning=tuning_report if isinstance(tuning_report, dict) else None,
            )

        except ValidationError as e:
            run.mark_failed(error=str(e))
            self.repo.save(run)
            raise
        except Exception as e:
            run.mark_failed(error=str(e))
            self.repo.save(run)
            raise TrainingFailed(f'Обучение (auto) не выполнено: {e}') from e
