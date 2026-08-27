from typing import Annotated

from fastapi import Depends

from inference.predictor import HousePricePredictor
from schemas import HealthResponse, HouseFeatures, PricePrediction

PredictorDep = Annotated[HousePricePredictor, Depends()]


class PredictionView:
    @staticmethod
    def health(predictor: PredictorDep) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=type(predictor.model).__name__,
            models_dir=str(predictor.models_dir),
        )

    @staticmethod
    def predict(features: HouseFeatures, predictor: PredictorDep) -> PricePrediction:
        return PricePrediction(
            price_pred=predictor.predict_single(features.model_dump()),
        )
