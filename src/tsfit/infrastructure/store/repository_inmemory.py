from __future__ import annotations

from tsfit.application.ports import TrainingRunRepository
from tsfit.domain.entites import TrainingRunEntity


class InMemoryTrainingRunRepository(TrainingRunRepository):
    '''
    Хранилище запусков обучения в памяти процесса # TODO: сделать нормально а не это
    '''

    def __init__(self) -> None:
        self._runs: dict[str, TrainingRunEntity] = {}

    def save(self, run: TrainingRunEntity) -> None:
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> TrainingRunEntity | None:
        return self._runs.get(run_id)

    def find_by_idempotency(self, key: str) -> TrainingRunEntity | None:
        for r in self._runs.values():
            if r.idempotency_key == key:
                return r
        return None
