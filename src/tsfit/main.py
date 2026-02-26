from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from dishka.integrations.fastapi import setup_dishka

from tsfit.presentation.routers import router as tsfit_router
from tsfit.bootstrap.di import container


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    Жизненный цикл приложения. D конце цикла закрываем контейнер dishka
    '''
    yield
    await app.state.dishka_container.close()


def create_app() -> FastAPI:
    '''
    Cоздание и настройка fastapi-приложения
    '''
    app = FastAPI(
        title='XGBoost Time Series train API',
        description='Cервис для обучения xgboost на временных рядах',
        version='0.2.0',
        lifespan=lifespan,
    )
    app.include_router(tsfit_router)

    setup_dishka(container=container, app=app)
    return app


app = create_app()
