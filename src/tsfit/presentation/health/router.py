from __future__ import annotations

from fastapi import APIRouter
from starlette import status as s


router = APIRouter(tags=['health'])


@router.get('/health', status_code=s.HTTP_200_OK)
async def health() -> dict[str, str]:
    return {'status': 'ok'}
