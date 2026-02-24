from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Callable
import uuid

from tsfit.application.ports import (
    TrainingRunRepository,
    DatasetParser,
    TimeSeriesDatasetBuilder,
    ModelTrainer,
    BuiltDataset,
)
from tsfit.domain.exceptions import IdempotencyConflict, TrainingFailed, ValidationError
from tsfit.domain.entities import TrainingRunEntity, RunStatus
from tsfit.domain.value_objects import (
    DatasetSchemaValueObject, 
    TimeSeriesConfigValueObject, 
    TrainingConfigValueObject,
    )
from tsfit.domain.rules import rule_validate_rows_have_columns
from tsfit.application.utils.payload_hashing import hash_fit_payload


@dataclass(frozen=True)
class TrainModelRequest:
    '''
    Запрос на обучение модели
    '''
    dataset_rows: list[dict[str, Any]]
    dataset_schema: DatasetSchemaValueObject
    time_series: TimeSeriesConfigValueObject
    training: TrainingConfigValueObject
    idempotency_key: str | None = None


@dataclass(frozen=True)
class TrainModelResult:
    '''
    Результат обучения модели
    '''
    run_id: str
    status: RunStatus
    created: bool
    metrics: dict[str, float] | None
    data_profile: dict[str, int] | None


class TrainModelUseCase:
    '''
    Сценарий для обучения модели XGBoost по временному ряду и возвращения метрик
    Последовательность:
    1) доменные провеки
    2) расчет хэша
    3) проверка идемпотентности по idempotency_key
    4) созданиеTrainingRunEntity
    5) парсинг данных и сборка датасета + препроцессинг
    6) обучение модели и расчет метрик
    7) сохранение в репозитр
    '''
    def __init__(
        self,
        repo: TrainingRunRepository,
        parser: DatasetParser,
        builder: TimeSeriesDatasetBuilder,
        trainer: ModelTrainer,
        id_gen: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repo = repo
        self.parser = parser
        self.builder = builder
        self.trainer = trainer
        self.id_gen = id_gen or (lambda: str(uuid.uuid4()))
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def execute(self, req: TrainModelRequest) -> TrainModelResult:

        # простые доменные проверки (инварианты)
        req.dataset_schema.validate()
        req.time_series.validate()
        req.training.validate()
        rule_validate_rows_have_columns(req.dataset_rows, req.dataset_schema)

        # хэш
        payload_hash = hash_fit_payload(
            dataset_rows=req.dataset_rows,
            dataset_schema=req.dataset_schema,
            time_series=req.time_series,
            training=req.training,
        )

        # идемпотентность
        
        if req.idempotency_key:
            existing = self.repo.find_by_idempotency(req.idempotency_key)
            if existing:
                if existing.payload_hash and existing.payload_hash != payload_hash:
                    raise IdempotencyConflict('Одинаковый idempotency_key, но разные данные/параметры')

                metrics: dict[str, float] | None = None
                if existing.result and isinstance(existing.result.get('metrics'), dict):
                    metrics = existing.result['metrics']

                data_profile: dict[str, int] | None = None
                if existing.result and isinstance(existing.result.get('data_profile'), dict):
                    dp = existing.result['data_profile']
                    if all(isinstance(v, int) for v in dp.values()):
                        data_profile = dp
                return TrainModelResult(
                    run_id=existing.run_id,
                    status=existing.status,
                    created=False,
                    metrics=metrics,
                    data_profile=data_profile,
                )

        # создание запуска
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

            # парсинг и строгая проверка временной оси
            frame = self.parser.parse(req.dataset_rows, req.dataset_schema)

            # сплит
            built: BuiltDataset = self.builder.build_train_valid(
                frame=frame,
                schema=req.dataset_schema,
                cfg=req.time_series,
            )

            # обучить и оценить метрики
            result = self.trainer.train_and_eval(dataset=built, training=req.training)


            # результат
            result.setdefault('feature_names', built.feature_names)

            data_profile = {
                'n_rows_raw': int(built.n_rows_raw),
                'n_total_after_features': int(built.n_total_after_features),
                'n_train': int(built.n_train),
                'n_valid': int(built.n_valid),
                'n_features': int(built.n_features),
            }
            result['data_profile'] = data_profile

            # фиксируется итоговая конфигурация обучения
            result['final_training'] = {
                'model_params': result.get('model_params'),
                'metrics': list(req.training.metrics),
                'primary_metric': str(req.training.primary_metric),
            }

            run.mark_done(result=result)
            self.repo.save(run)

            metrics = result.get('metrics')
            return TrainModelResult(
                run_id=run.run_id,
                status=run.status,
                created=True,
                metrics=metrics if isinstance(metrics, dict) else None,
                data_profile=data_profile,
            )

        except ValidationError as e:
            run.mark_failed(error=str(e))
            self.repo.save(run)
            raise
        except Exception as e:
            run.mark_failed(error=str(e))
            self.repo.save(run)
            raise TrainingFailed(f'Обучение не выполнено: {e}') from e
        


    