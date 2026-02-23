from tsfit.domain.entities import TrainingRunEntity
from tsfit.application.ports import TrainingRunRepository


class GetRunUseCase:
    '''
    Сценарий для получения информации о runе по его айди
    '''
    def __init__(self, repo: TrainingRunRepository) -> None:
        self.repo = repo

    def execute(self, run_id: str) -> TrainingRunEntity | None:
        return self.repo.get(run_id)