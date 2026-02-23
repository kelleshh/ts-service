from fastapi import (
    APIRouter,
    Request,
    Depends,
    Response,
    status as s,
    HTTPException
)

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from starlette.concurrency import run_in_threadpool # чтобы не блокировать event loop при выполнении usecase по обучению модели

from tsfit.api.schemas import FitRequest, FitResponse
from tsfit.application.usecases import TrainModelUseCase, TrainModelRequest, GetRunUseCase
from tsfit.domain.exceptions import IdempotencyConflict, ValidationError, TrainingFailed

router = APIRouter(route_class=DishkaRoute)

@router.post('/fit', response_model=FitResponse)
async def fit(
    req: FitRequest,
    response: Response,
    uc: FromDishka[TrainModelUseCase],
) -> FitResponse:
    try:
        usecase_req = TrainModelRequest(
            dataset_rows=req.dataset,
            dataset_schema=req.dataset_schema.to_domain(),
            time_series=req.ts.to_domain(),
            training=req.training.to_domain(),
            idempotency_key=req.idempotency_key,
        )
        result = await run_in_threadpool(uc.execute, usecase_req)

    except IdempotencyConflict as e:
        raise HTTPException(status_code=s.HTTP_409_CONFLICT, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=s.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except TrainingFailed as e:
        # внутренняя ошибка обучения/пайплайна
        raise HTTPException(status_code=s.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


    response.status_code = s.HTTP_201_CREATED if result.created else s.HTTP_200_OK

    return FitResponse(
        run_id=result.run_id,
        status=result.status,
        created=result.created,
        metrics=result.metrics,
    )


@router.get('/runs/{run_id}')
async def get_run(
    run_id: str, 
    uc: FromDishka[GetRunUseCase]
    ) -> dict:

    run = uc.execute(run_id)
    if run is None:
        raise HTTPException(status_code=s.HTTP_404_NOT_FOUND, detail='Run не найден')

    return {
        'run_id': run.run_id,
        'status': run.status.value,
        'created_at': run.created_at.isoformat(),
        'error': run.error,
        'idempotency_key': run.idempotency_key,
    }


@router.get('/runs/{run_id}/result')
async def get_run_result(
    run_id: str, 
    uc: FromDishka[GetRunUseCase]
    ) -> dict:

    run = uc.execute(run_id)
    if run is None:
        raise HTTPException(status_code=s.HTTP_404_NOT_FOUND, detail='Run не найден')

    return {
        'run_id': run.run_id,
        'result': run.result,
    }