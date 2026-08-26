"""House price prediction API — entry point."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import Settings
from deps import PredictorService
from routers.prediction import PredictionRouter


class Application:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self._app = FastAPI(
            title=self.settings.api_title,
            version=self.settings.api_version,
            lifespan=self._lifespan,
        )
        self._configure_cors()
        self._register_routers()

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI):
        PredictorService.init(self.settings)
        yield

    def _configure_cors(self):
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routers(self):
        self._app.include_router(PredictionRouter.router)

    @property
    def app(self) -> FastAPI:
        return self._app


if __name__ == "__main__":
    uvicorn.run("app:Application().app", factory=True, host="0.0.0.0", port=8000, reload=True)
