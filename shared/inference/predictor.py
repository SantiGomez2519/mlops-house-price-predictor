"""Single source of truth for raw → prediction inference."""

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn import set_config

from transformers import custom_transformers
from transformers.custom_transformers import AddHouseAge  # noqa: F401

sys.modules["custom_transformers"] = custom_transformers

set_config(transform_output="pandas")

TARGET = "price"


class HousePricePredictor:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.preprocessor = None
        self.feature_engineer = None
        self.model = None
        self._load()

    def _load(self):
        with open(self.models_dir / "preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(self.models_dir / "feature_engineer.pkl", "rb") as f:
            self.feature_engineer = pickle.load(f)
        with open(self.models_dir / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.preprocessor.set_output(transform="pandas")
        self.feature_engineer.named_steps["encode"].set_output(transform="pandas")

    def _prepare_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure raw dataframe has the dummy target column the preprocessor expects."""
        if TARGET not in df.columns:
            df = df.copy()
            df[TARGET] = 0.0
        return df

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Raw DataFrame → DataFrame with original columns + price_pred."""
        prepared = self._prepare_raw(raw_df)
        preprocessed = self.preprocessor.transform(prepared)
        featured = self.feature_engineer.transform(preprocessed)
        if TARGET in featured.columns:
            featured = featured.drop(columns=[TARGET])
        featured = featured[list(self.model.feature_names_in_)]
        preds = self.model.predict(featured)
        result = raw_df.copy()
        result["price_pred"] = preds
        return result

    def predict_single(self, features: dict) -> float:
        """Single dict of raw features → predicted price."""
        raw_df = pd.DataFrame([features])
        result = self.predict_batch(raw_df)
        return float(result["price_pred"].iloc[0])
