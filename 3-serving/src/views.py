"""Prediction views — class-based, DRF style."""

from fastapi import Request

from schemas import (
    HousePriceFeatures,
    HousePriceHealthResponse,
    HousePricePrediction,
)


class HousePricePredictionView:
    @staticmethod
    def health(request: Request) -> HousePriceHealthResponse:
        predictor = request.app.state.predictor
        return HousePriceHealthResponse(
            status="ok",
            model=type(predictor.model).__name__,
            models_dir=str(predictor.models_dir),
        )

    @staticmethod
    def predict(
        features: HousePriceFeatures,
        request: Request,
    ) -> HousePricePrediction:
        predictor = request.app.state.predictor
        return HousePricePrediction(
            price_pred=predictor.predict_single(features.model_dump()),
        )
