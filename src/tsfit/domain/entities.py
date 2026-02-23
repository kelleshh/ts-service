from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from tsfit.domain.exceptions import InvalidRunTransition

class RunStatus(str, Enum):
    '''
    Статусы запуска
    '''
    # TODO: добавить QUEUED
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    DONE = 'DONE'
    FAILED = 'FAILED'

@dataclass
class TrainingRunEntity:
    run_id: str
    status: RunStatus
    created_at: datetime

    idempotency_key: str | None = None
    payload_hash: str | None = None

    error: str | None = None
    result: dict[str, Any] | None = None

    @classmethod
    def new_pending( # автоматическая простановка статуса Pending
        cls,
        run_id: str,
        created_at: datetime,
        *,
        idempotency_key: str | None,
        payload_hash: str | None,
    ) -> 'TrainingRunEntity':
        return cls(
            run_id=run_id,
            status=RunStatus.PENDING,
            created_at=created_at,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
        )
    def mark_running(self) -> None:
        if self.status != RunStatus.PENDING:
            raise InvalidRunTransition(
                f'Нельзя mark_running из {self.status}'
                )
        self.status = RunStatus.RUNNING
    
    def mark_done(self, result: dict[str, Any]) -> None:
        if self.status != RunStatus.RUNNING:
            raise InvalidRunTransition(f'Нельзя mark_done из {self.status}')
        self.status = RunStatus.DONE
        self.result = result
        self.error = None

    def mark_failed(self, error: str) -> None:
        if self.status not in (RunStatus.PENDING, RunStatus.RUNNING):
            raise InvalidRunTransition(
                'Нельзя mark_failed из DONE'
            )
        self.status = RunStatus.FAILED
        self.error = error