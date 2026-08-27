"""House price prediction API — entry point."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import HousePriceSettings
from inference.predictor import HousePricePredictor
from urls import HousePriceRouter


class HousePriceApplication:
    def __init__(self):
        self._settings = HousePriceSettings.from_env()
        self._app = FastAPI(
            title=self._settings.api_title,
            version=self._settings.api_version,
            lifespan=self._lifespan,
        )
        self._app.state.settings = self._settings
        self._configure_cors()
        self._register_routes()

    @staticmethod
    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        settings: HousePriceSettings = _app.state.settings
        _app.state.predictor = HousePricePredictor(settings.models_dir)
        yield

    def _configure_cors(self) -> None:
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=self._settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self) -> None:
        self._app.include_router(HousePriceRouter.register())

    @property
    def app(self) -> FastAPI:
        return self._app


app = HousePriceApplication().app

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
