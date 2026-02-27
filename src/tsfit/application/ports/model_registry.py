from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.domain.value_objects.meta import ModelMeta
from tsfit.application.results import TrainingReport


# Port
class ModelRegistry(ABC):
    @abstractmethod
    def find_by_idempotency_key(self, idempotency_key: str) -> tuple[str, str] | None:
        '''
        Возвращает (model_id, payload_hash) или None
        '''
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        *,
        idempotency_key: str,
        payload_hash: str,
        report: TrainingReport,
        meta: ModelMeta,
        summary: dict,
        extra: dict[str, object],
    ) -> str:
        '''
        Возвращает model_id
        '''
        raise NotImplementedError

    @abstractmethod
    def load_model_and_meta(self, model_id: str) -> tuple[object, ModelMeta]:
        raise NotImplementedError

    @abstractmethod
    def load_summary(self, model_id: str) -> dict:
        raise NotImplementedError
