from fastapi import FastAPI
from dishka.integrations.fastapi import setup_dishka

from tsfit.api.routes import router as tsfit_router
from tsfit.bootstrap.di import container

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    Жизненный цикл приложения. В конце цикла закрываем контейнер dishka
    '''
    yield
    await app.state.dishka_container.close()

def create_app() -> FastAPI:
    '''
    Создание и настройка FastAPI-приложения
    '''
    app = FastAPI(
        title='XGBoost Time Series train API',
        description='Сервис для обучения XGBoost на временных рядах',
        version='0.1.2',
        lifespan=lifespan
    )
    app.include_router(tsfit_router)

    setup_dishka(container=container, app=app)
    return app


app = create_app()