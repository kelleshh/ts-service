from __future__ import annotations

from tsfit.application.ports import TrainingRunRepository
from tsfit.domain.training_run import TrainingRun


class InMemoryTrainingRunRepository(TrainingRunRepository):
    '''
    Хранилище запусков обучения в памяти процесса # TODO: сделать нормально а не это
    '''

    def __init__(self) -> None:
        self._runs: dict[str, TrainingRun] = {}

    def save(self, run: TrainingRun) -> None:
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> TrainingRun | None:
        return self._runs.get(run_id)

    def find_by_idempotency(self, key: str) -> TrainingRun | None:
        for r in self._runs.values():
            if r.idempotency_key == key:
                return r
        return None
