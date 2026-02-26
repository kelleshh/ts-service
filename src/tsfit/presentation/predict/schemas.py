from typing import Any
from pydantic import Field

from tsfit.presentation.apimodel import APIModel

class PredictRequest(APIModel):
    model_id: str = Field(min_length=1)
    dataset: list[dict[str, Any]] = Field(min_length=1)


class PredictResponse(APIModel):
    prediction: float
