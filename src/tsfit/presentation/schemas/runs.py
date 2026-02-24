from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from tsfit.domain.entities import RunStatus


class RunInfoResponse(BaseModel):
    '''
    Ответ для GET /runs/{run_id}
    '''

    model_config = ConfigDict(extra='forbid')

    run_id: str
    status: RunStatus
    created_at: str
    error: str | None
    idempotency_key: str | None


class RunResultResponse(BaseModel):
    '''
    Ответ для GET /runs/{run_id}/result
    '''

    model_config = ConfigDict(extra='forbid')

    run_id: str
    result: dict[str, Any] | None
