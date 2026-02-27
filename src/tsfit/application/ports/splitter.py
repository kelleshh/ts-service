from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.application.types import DatasetHandle, Fold
from tsfit.domain.value_objects import CVPlan


# Port
class WalkForwardSplitter(ABC):
    @abstractmethod
    def split(self, supervised: DatasetHandle, plan: CVPlan) -> list[Fold]:
        raise NotImplementedError
