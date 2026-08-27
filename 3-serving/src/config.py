"""Application configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path


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
        src_dir = Path(__file__).resolve().parent
        repo_dir = src_dir.parent.parent
        local_models = src_dir / "models"
        industrial_models = repo_dir / "2-industrialization" / "src" / "models"

        env_dir = os.environ.get("MODELS_DIR")
        if env_dir:
            models_dir = Path(env_dir)
        elif (local_models / "model.pkl").exists():
            models_dir = local_models
        else:
            models_dir = industrial_models
        return cls(models_dir=models_dir)
