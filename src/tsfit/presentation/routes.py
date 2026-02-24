from fastapi import APIRouter

from dishka.integrations.fastapi import DishkaRoute

from tsfit.presentation.routers.health import router as health_router
from tsfit.presentation.routers.runs import router as runs_router


router = APIRouter(route_class=DishkaRoute)
router.include_router(health_router)
router.include_router(runs_router)