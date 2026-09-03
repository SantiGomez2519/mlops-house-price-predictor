"""House price prediction API — single-file serving."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.predictor import HousePricePredictor

MODELS_DIR = Path(
    os.environ.get("MODELS_DIR")
    or (Path(__file__).resolve().parent.parent.parent / "shared" / "models")
)

DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


class HousePriceFeatures(BaseModel):
    sqft: float = Field(examples=[1527])
    bedrooms: float = Field(examples=[2])
    bathrooms: float = Field(examples=[1.5])
    location: str = Field(examples=["Suburb"])
    year_built: float = Field(examples=[1956])
    condition: str = Field(examples=["Good"])


class HousePricePrediction(BaseModel):
    price_pred: float


class HousePriceHealthResponse(BaseModel):
    status: str
    model: str
    models_dir: str


class HousePricePredictionView:
    @staticmethod
    def health() -> HousePriceHealthResponse:
        return HousePriceHealthResponse(
            status="ok",
            model=type(predictor.model).__name__,
            models_dir=str(predictor.models_dir),
        )

    @staticmethod
    def predict(features: HousePriceFeatures) -> HousePricePrediction:
        return HousePricePrediction(
            price_pred=predictor.predict_single(features.model_dump()),
        )


class HousePriceRouter:
    router = APIRouter()

    @classmethod
    def register(cls) -> APIRouter:
        cls.router.add_api_route(
            "/health",
            HousePricePredictionView.health,
            methods=["GET"],
        )
        cls.router.add_api_route(
            "/predict",
            HousePricePredictionView.predict,
            methods=["POST"],
        )
        return cls.router


predictor = None


class HousePriceApplication:
    def __init__(self):
        self._app = FastAPI(title="House price predictor", lifespan=self._lifespan)
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=DEFAULT_CORS_ORIGINS,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._app.include_router(HousePriceRouter.register())

    @staticmethod
    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        global predictor
        predictor = HousePricePredictor(MODELS_DIR)
        yield

    @property
    def app(self) -> FastAPI:
        return self._app


app = HousePriceApplication().app

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
