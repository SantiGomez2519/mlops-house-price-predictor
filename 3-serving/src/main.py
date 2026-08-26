"""House price prediction API."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.predictor import HousePricePredictor


class HouseFeatures(BaseModel):
    sqft: float = Field(examples=[1527])
    bedrooms: float = Field(examples=[2])
    bathrooms: float = Field(examples=[1.5])
    location: str = Field(examples=["Suburb"])
    year_built: float = Field(examples=[1956])
    condition: str = Field(examples=["Good"])


class PricePrediction(BaseModel):
    price_pred: float


class HealthResponse(BaseModel):
    status: str
    model: str
    models_dir: str


SRC_DIR = Path(__file__).resolve().parent
SERVING_DIR = SRC_DIR.parent
REPO_DIR = SERVING_DIR.parent
LOCAL_MODELS_DIR = SRC_DIR / "models"
INDUSTRIAL_MODELS_DIR = REPO_DIR / "2-industrialization" / "src" / "models"

predictor: HousePricePredictor | None = None
models_dir: Path | None = None


def resolve_models_dir() -> Path:
    env_dir = os.environ.get("MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    if (LOCAL_MODELS_DIR / "model.pkl").exists():
        return LOCAL_MODELS_DIR
    return INDUSTRIAL_MODELS_DIR


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global predictor, models_dir
    try:
        models_dir = resolve_models_dir()
        predictor = HousePricePredictor(models_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Model artifacts not found. Run 2-industrialization first, "
            "or set MODELS_DIR to a folder with preprocessor.pkl, "
            "feature_engineer.pkl, and model.pkl."
        ) from exc
    yield


app = FastAPI(title="House price predictor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model=type(predictor.model).__name__,
        models_dir=str(models_dir),
    )


@app.post("/predict", response_model=PricePrediction)
def predict(features: HouseFeatures) -> PricePrediction:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return PricePrediction(price_pred=predictor.predict_single(features.model_dump()))
