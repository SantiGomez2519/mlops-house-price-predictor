"""Application configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path


class _Paths:
    SRC_DIR = Path(__file__).resolve().parent
    SERVING_DIR = SRC_DIR.parent
    REPO_DIR = SERVING_DIR.parent
    LOCAL_MODELS_DIR = SRC_DIR / "models"
    INDUSTRIAL_MODELS_DIR = REPO_DIR / "2-industrialization" / "src" / "models"


DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    cors_origins: list[str] = field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    api_title: str = "House price predictor"
    api_version: str = "0.1.0"

    @classmethod
    def from_env(cls) -> "Settings":
        env_dir = os.environ.get("MODELS_DIR")
        if env_dir:
            models_dir = Path(env_dir)
        elif (_Paths.LOCAL_MODELS_DIR / "model.pkl").exists():
            models_dir = _Paths.LOCAL_MODELS_DIR
        else:
            models_dir = _Paths.INDUSTRIAL_MODELS_DIR
        return cls(models_dir=models_dir)
