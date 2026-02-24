from __future__ import annotations

from fastapi import APIRouter, HTTPException, status as s

from dishka.integrations.fastapi import DishkaRoute, FromDishka

from tsfit.application.usecases.get_usecase import GetRunUseCase
from tsfit.presentation.schemas.runs import RunInfoResponse, RunResultResponse


router = APIRouter(prefix='/runs', route_class=DishkaRoute, tags=['runs'])


@router.get('/{run_id}', response_model=RunInfoResponse)
async def get_run(run_id: str, uc: FromDishka[GetRunUseCase]) -> RunInfoResponse:
    run = uc.execute(run_id)
    if not run:
        raise HTTPException(status_code=s.HTTP_404_NOT_FOUND, detail='run не найден')

    return RunInfoResponse(
        run_id=run.run_id,
        status=run.status,
        created_at=run.created_at.isoformat(),
        error=run.error,
        idempotency_key=run.idempotency_key,
    )


@router.get('/{run_id}/result', response_model=RunResultResponse)
async def get_run_result(run_id: str, uc: FromDishka[GetRunUseCase]) -> RunResultResponse:
    run = uc.execute(run_id)
    if not run:
        raise HTTPException(status_code=s.HTTP_404_NOT_FOUND, detail='run не найден')

    return RunResultResponse(run_id=run.run_id, result=run.result)
