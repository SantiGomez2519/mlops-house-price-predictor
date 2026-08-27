"""House price prediction API — entry point."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import Settings
from inference.predictor import HousePricePredictor

predictor: HousePricePredictor | None = None


def get_predictor() -> HousePricePredictor:
    if predictor is None:
        raise RuntimeError("Model not loaded")
    return predictor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global predictor
    settings = Settings.from_env()
    predictor = HousePricePredictor(settings.models_dir)
    yield


def create_app() -> FastAPI:
    settings = Settings.from_env()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )
    app.dependency_overrides[HousePricePredictor] = get_predictor
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from urls import router
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
