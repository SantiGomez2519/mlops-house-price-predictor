"""FastAPI dependencies — model loading and injection."""

from config import Settings
from inference.predictor import HousePricePredictor


class PredictorService:
    """Wraps HousePricePredictor for FastAPI dependency injection."""

    _instance: "PredictorService | None" = None

    def __init__(self, settings: Settings):
        self._settings = settings
        self._predictor = HousePricePredictor(settings.models_dir)

    def predict_single(self, features: dict) -> float:
        return self._predictor.predict_single(features)

    def predict_batch(self, raw_df):
        return self._predictor.predict_batch(raw_df)

    @property
    def model_name(self) -> str:
        return type(self._predictor.model).__name__

    @property
    def models_dir(self) -> str:
        return str(self._settings.models_dir)

    @classmethod
    def init(cls, settings: Settings) -> "PredictorService":
        cls._instance = cls(settings)
        return cls._instance

    @classmethod
    def get(cls) -> "PredictorService":
        if cls._instance is None:
            raise RuntimeError("PredictorService not initialized. Call .init() first.")
        return cls._instance
