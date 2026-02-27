from __future__ import annotations

import uuid

from tsfit.domain.value_objects.meta import ModelMeta
from tsfit.application.ports import ModelRegistry
from tsfit.application.results import TrainingReport
from tsfit.domain.exceptions import NotFoundError


# Adapter (Infrastructure)
class InMemoryModelRegistry(ModelRegistry):
    def __init__(self) -> None:
        self._by_idempotency: dict[str, tuple[str, str]] = {}
        self._models: dict[str, tuple[object, ModelMeta, dict]] = {}

    def find_by_idempotency_key(self, idempotency_key: str) -> tuple[str, str] | None:
        return self._by_idempotency.get(idempotency_key)

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
        model_id = str(uuid.uuid4())
        self._by_idempotency[idempotency_key] = (model_id, payload_hash)
        self._models[model_id] = (report.model, meta, summary)
        return model_id

    def load_model_and_meta(self, model_id: str) -> tuple[object, ModelMeta]:
        if model_id not in self._models:
            raise NotFoundError('model_id не найден')
        model, meta, _ = self._models[model_id]
        return model, meta

    def load_summary(self, model_id: str) -> dict:
        if model_id not in self._models:
            raise NotFoundError('model_id не найден')
        _, _, summary = self._models[model_id]
        return dict(summary)
