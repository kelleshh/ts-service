from __future__ import annotations

from fastapi import APIRouter

from dishka.integrations.fastapi import DishkaRoute

from tsfit.presentation.health.router import router as health_router
from tsfit.presentation.predict.router import router as predict_router
from tsfit.presentation.training.router import router as training_router


router = APIRouter(route_class=DishkaRoute)
router.include_router(health_router)
router.include_router(training_router)
router.include_router(predict_router)
