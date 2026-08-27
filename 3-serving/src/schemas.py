"""Pydantic schemas for request / response validation."""

from pydantic import BaseModel, Field


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
