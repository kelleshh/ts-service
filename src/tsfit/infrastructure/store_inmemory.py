from __future__ import annotations
from threading import Lock
from typing import Dict

from tsfit.application.ports import TrainingRunRepository
from tsfit.domain.training_run import TrainingRun
from tsfit.domain.errors import IdempotencyConflict


class InMemoryTrainingRunRepository(TrainingRunRepository):
    def __init__(self) -> None:
        self._runs: Dict[str, TrainingRun] = {}
        self._idem: Dict[str, str] = {}  # idempotency_key -> run_id
        self._lock = Lock()

    def save(self, run: TrainingRun) -> None:
        with self._lock:
            if run.idempotency_key:
                existing_run_id = self._idem.get(run.idempotency_key)
                if existing_run_id and existing_run_id != run.run_id:
                    raise IdempotencyConflict('ключ idempotency_key уже занят другим runом')
                self._idem[run.idempotency_key] = run.run_id

            self._runs[run.run_id] = run

    def get(self, run_id: str) -> TrainingRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def find_by_idempotency(self, key: str) -> TrainingRun | None:
        with self._lock:
            run_id = self._idem.get(key)
            return self._runs.get(run_id) if run_id else None