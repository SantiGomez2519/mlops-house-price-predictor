from schemas import (
    HousePriceFeatures,
    HousePriceHealthResponse,
    HousePricePrediction,
)

predictor = None

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
