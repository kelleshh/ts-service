from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.domain.value_objects import DatasetSchema, TrainingConfig


# Port
class PayloadHasher(ABC):
    @abstractmethod
    def hash_fit_payload(
        self,
        *,
        rows: list[dict],
        schema: DatasetSchema,
        horizon: int,
        training: TrainingConfig,
        tuning: dict[str, object] | None,
        policy_version: str,
    ) -> str:
        raise NotImplementedError
