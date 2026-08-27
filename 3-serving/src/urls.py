"""URL routing — maps endpoints to views."""

from fastapi import APIRouter

from views import PredictionView

router = APIRouter()
router.add_api_route("/health", PredictionView.health, methods=["GET"])
router.add_api_route("/predict", PredictionView.predict, methods=["POST"])
