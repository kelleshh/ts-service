from fastapi import APIRouter

from dishka.integrations.fastapi import DishkaRoute

from tsfit.presentation.routers.health import (
    router as health_router,
)


router = APIRouter(route_class=DishkaRoute)
router.include_router(health_router)