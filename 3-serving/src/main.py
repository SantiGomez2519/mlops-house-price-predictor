"""House price prediction API."""

import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn import set_config

set_config(transform_output="pandas")


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


class HousePriceServing:
    TARGET = "price"

    SRC_DIR = Path(__file__).resolve().parent
    SERVING_DIR = SRC_DIR.parent
    REPO_DIR = SERVING_DIR.parent
    LOCAL_MODELS_DIR = SRC_DIR / "models"
    INDUSTRIAL_MODELS_DIR = REPO_DIR / "2-industrialization" / "src" / "models"
    PIPELINE_DIR = REPO_DIR / "2-industrialization" / "src" / "pipeline"

    preprocessor = None
    feature_engineer = None
    model = None
    models_dir: Path | None = None

    @classmethod
    def resolve_models_dir(cls) -> Path:
        env_dir = os.environ.get("MODELS_DIR")
        if env_dir:
            return Path(env_dir)
        if (cls.LOCAL_MODELS_DIR / "model.pkl").exists():
            return cls.LOCAL_MODELS_DIR
        return cls.INDUSTRIAL_MODELS_DIR

    @classmethod
    def load(cls) -> None:
        if str(cls.PIPELINE_DIR) not in sys.path:
            sys.path.insert(0, str(cls.PIPELINE_DIR))
        from custom_transformers import AddHouseAge  # noqa: F401

        cls.models_dir = cls.resolve_models_dir()
        with open(cls.models_dir / "preprocessor.pkl", "rb") as file:
            cls.preprocessor = pickle.load(file)
        with open(cls.models_dir / "feature_engineer.pkl", "rb") as file:
            cls.feature_engineer = pickle.load(file)
        with open(cls.models_dir / "model.pkl", "rb") as file:
            cls.model = pickle.load(file)
        cls.preprocessor.set_output(transform="pandas")
        cls.feature_engineer.named_steps["encode"].set_output(transform="pandas")

    @classmethod
    def predict(cls, features: HouseFeatures) -> float:
        raw_df = pd.DataFrame([features.model_dump()])
        raw_df[cls.TARGET] = 0.0
        preprocessed = cls.preprocessor.transform(raw_df)
        featured = cls.feature_engineer.transform(preprocessed)
        if cls.TARGET in featured.columns:
            featured = featured.drop(columns=[cls.TARGET])
        featured = featured[list(cls.model.feature_names_in_)]
        return float(cls.model.predict(featured)[0])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        HousePriceServing.load()
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
    if HousePriceServing.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model=type(HousePriceServing.model).__name__,
        models_dir=str(HousePriceServing.models_dir),
    )


@app.post("/predict", response_model=PricePrediction)
def predict(features: HouseFeatures) -> PricePrediction:
    if HousePriceServing.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return PricePrediction(price_pred=HousePriceServing.predict(features))
