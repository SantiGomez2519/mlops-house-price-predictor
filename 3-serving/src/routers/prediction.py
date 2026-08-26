"""Prediction routes: /health and /predict."""

from fastapi import APIRouter, Depends

from deps import PredictorService
from schemas import HealthResponse, HouseFeatures, PricePrediction


class PredictionRouter:
    router = APIRouter()

    @staticmethod
    @router.get("/health", response_model=HealthResponse)
    def health(
        service: PredictorService = Depends(PredictorService.get),
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=service.model_name,
            models_dir=service.models_dir,
        )

    @staticmethod
    @router.post("/predict", response_model=PricePrediction)
    def predict(
        features: HouseFeatures,
        service: PredictorService = Depends(PredictorService.get),
    ) -> PricePrediction:
        return PricePrediction(price_pred=service.predict_single(features.model_dump()))
