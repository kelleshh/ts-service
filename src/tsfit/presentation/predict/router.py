from __future__ import annotations

from fastapi import APIRouter, Response, HTTPException
from starlette import status as s

from dishka.integrations.fastapi import DishkaRoute, FromDishka

from tsfit.application.usecases.commands import PredictCommand
from tsfit.application.usecases.predict import PredictUseCase

from tsfit.presentation.predict.schemas import (
    PredictRequest,
    PredictResponse,
)

from tsfit.domain.exceptions import ConflictError, ValidationError, InvariantError, NotFoundError




router = APIRouter(route_class=DishkaRoute, tags=['predict'], prefix='/predict')


@router.post('/', status_code=s.HTTP_200_OK, response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    response: Response,
    uc: FromDishka[PredictUseCase],
) -> PredictResponse:
    try:
        cmd = PredictCommand(model_id=req.model_id, rows=req.dataset)
        result = uc.execute(cmd)
        response.status_code = s.HTTP_200_OK
        return PredictResponse(prediction=float(result.prediction))
    
    except ValidationError as e:
        raise HTTPException(status_code=s.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except (InvariantError, ConflictError) as e:
        raise HTTPException(status_code=s.HTTP_409_CONFLICT, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=s.HTTP_404_NOT_FOUND, detail=str(e))

