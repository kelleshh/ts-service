from __future__ import annotations

from abc import ABC, abstractmethod

from tsfit.application.types import DatasetHandle


# Port
class DatasetParser(ABC):
    @abstractmethod
    def parse(self, rows: list[dict]) -> DatasetHandle:
        raise NotImplementedError
