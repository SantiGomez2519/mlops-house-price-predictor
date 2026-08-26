"""Pydantic schemas for request / response validation."""

from pydantic import BaseModel, Field


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
