from fastapi import APIRouter

from views import HousePricePredictionView


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
