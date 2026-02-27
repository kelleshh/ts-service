from __future__ import annotations

from abc import ABC, abstractmethod


# Port
class Predictor(ABC):
    @abstractmethod
    def predict_one(self, model: object, x: object) -> float:
        raise NotImplementedError
