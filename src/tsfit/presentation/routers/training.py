from __future__ import annotations

from fastapi import (
    APIRouter,
    Response,
    status as s,
    HTTPException
)
from starlette.concurrency import run_in_threadpool

from dishka.integrations.fastapi import DishkaRoute, FromDishka

from tsfit.application.usecases.fit_usecase import (
    TrainModelRequest, 
    TrainModelUseCase
)
from tsfit.application.usecases.autofit_usecase import (
    TrainModelAutoRequest,
    TrainModelAutoUseCase,
)
from tsfit.domain.exceptions import (
    ValidationError, 
    IdempotencyConflict, 
    TrainingFailed
)
from tsfit.presentation.schemas.training import (
    FitRequest,
    FitResponse,
    FitAutoRequest,
    FitAutoResponse,
)


router = APIRouter(route_class=DishkaRoute, tags=['training'], prefix='/fit')


@router.post('/', status_code=s.HTTP_201_CREATED, response_model=FitResponse)
async def fit(
    req: FitRequest,
    response: Response,
    uc: FromDishka[TrainModelUseCase],
) -> FitResponse:
    '''
    Обучение модели по фиксированным параметрам без использования автотюнинга
    '''
    try:
        usecase_req = TrainModelRequest(
            dataset_rows=req.dataset,
            dataset_schema=req.dataset_schema.to_domain(),
            time_series=req.ts.to_domain(),
            training=req.training.to_domain(),
            idempotency_key=req.idempotency_key,
        )
        res = await run_in_threadpool(uc.execute, usecase_req)
        response.status_code = s.HTTP_201_CREATED if res.created else s.HTTP_200_OK
        return FitResponse(
            run_id=res.run_id,
            status=res.status,
            created=res.created,
            metrics=res.metrics,
            data_profile=res.data_profile,
        )
    except IdempotencyConflict as e:
        raise HTTPException(status_code=s.HTTP_409_CONFLICT, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=s.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except TrainingFailed as e:
        # внутренняя ошибка обучения/пайплайна
        raise HTTPException(status_code=s.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post('/auto', status_code=s.HTTP_201_CREATED, response_model=FitAutoResponse)
async def fit_auto(
    req: FitAutoRequest,
    response: Response,
    uc: FromDishka[TrainModelAutoUseCase],
) -> FitAutoResponse:
    '''
    Обучение модели с автоподбором гиперпараметров
    '''
    try:
        usecase_req = TrainModelAutoRequest(
            dataset_rows=req.dataset,
            dataset_schema=req.dataset_schema.to_domain(),
            time_series=req.ts.to_domain(),
            training=req.training.to_domain(),
            tuning=req.tuning.to_domain(),
            idempotency_key=req.idempotency_key,
        )
        res = await run_in_threadpool(uc.execute, usecase_req)
        response.status_code = s.HTTP_201_CREATED if res.created else s.HTTP_200_OK
        return FitAutoResponse(
            run_id=res.run_id,
            status=res.status,
            created=res.created,
            metrics=res.metrics,
            data_profile=res.data_profile,
            tuning_report=res.tuning_report,
        )
    except IdempotencyConflict as e:
        raise HTTPException(status_code=s.HTTP_409_CONFLICT, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=s.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except TrainingFailed as e:
        raise HTTPException(status_code=s.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
