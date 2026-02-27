from __future__ import annotations


from fastapi import APIRouter, Response, HTTPException
from starlette import status as s

from dishka.integrations.fastapi import DishkaRoute, FromDishka

from tsfit.application.usecases.train_model import TrainModelUseCase
from tsfit.application.usecases.train_model_auto import TrainModelAutoUseCase
from tsfit.application.usecases.commands import FitAutoCommand, FitCommand
from tsfit.domain.exceptions import ConflictError, ValidationError, InvariantError, NotFoundError

from tsfit.presentation.training.schemas import (
    FitRequest, FitResponse,
    FitAutoRequest, FitAutoResponse,
)
from starlette.concurrency import run_in_threadpool





router = APIRouter(route_class=DishkaRoute, tags=['training'], prefix='/fit')


@router.post('/', status_code=s.HTTP_201_CREATED, response_model=FitResponse)
async def fit(
    req: FitRequest,
    response: Response,
    uc: FromDishka[TrainModelUseCase],
) -> FitResponse:
    try:
        cmd = FitCommand(
            idempotency_key=req.idempotency_key,
            rows=req.dataset,
            timestamp_col=req.dataset_schema.timestamp_col,
            target_col=req.dataset_schema.target_col,
            exog_cols=req.dataset_schema.exog_cols,
            horizon=req.horizon,
            primary_metric=req.training.primary_metric,
            metrics=req.training.metrics,
            model_params=req.training.model_params,
            early_stopping_rounds=req.training.early_stopping_rounds,
            n_estimators_cap=req.training.n_estimators_cap,
        )

        result = await run_in_threadpool(uc.execute, cmd)
        response.status_code = s.HTTP_201_CREATED if result.created else s.HTTP_200_OK
        return FitResponse(model_id=result.model_id, created=result.created, summary=dict(result.summary))
    
    except ValidationError as e:
        raise HTTPException(status_code=s.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except (InvariantError, ConflictError) as e:
        raise HTTPException(status_code=s.HTTP_409_CONFLICT, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=s.HTTP_404_NOT_FOUND, detail=str(e))

@router.post('/auto', status_code=s.HTTP_201_CREATED, response_model=FitAutoResponse)
async def fit_auto(
    req: FitAutoRequest,
    response: Response,
    uc: FromDishka[TrainModelAutoUseCase],
) -> FitAutoResponse:
    try:
        cmd = FitAutoCommand(
        idempotency_key=req.idempotency_key,
        rows=req.dataset,
        timestamp_col=req.dataset_schema.timestamp_col,
        target_col=req.dataset_schema.target_col,
        exog_cols=req.dataset_schema.exog_cols,
        horizon=req.horizon,
        primary_metric=req.training.primary_metric,
        metrics=req.training.metrics,
        model_params=req.training.model_params,
        early_stopping_rounds=req.training.early_stopping_rounds,
        n_estimators_cap=req.training.n_estimators_cap,
        n_trials=req.tuning.n_trials,
        timeout_sec=req.tuning.timeout_sec

        )

        result = await run_in_threadpool(uc.execute, cmd)
        response.status_code = s.HTTP_201_CREATED if result.created else s.HTTP_200_OK
        return FitAutoResponse(model_id=result.model_id, created=result.created, summary=dict(result.summary))

    except ValidationError as e:
        raise HTTPException(status_code=s.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except (InvariantError, ConflictError) as e:
        raise HTTPException(status_code=s.HTTP_409_CONFLICT, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=s.HTTP_404_NOT_FOUND, detail=str(e))

