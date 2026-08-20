"""2.3 Feature engineering.

Fit house_age + scaling/encoding on preprocessed train (experimentation 1.4).
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from custom_transformers import AddHouseAge  # noqa: E402

set_config(transform_output="pandas")


class DataPreprocessedTrainFeatureEngineering:
    TARGET = "price"
    REFERENCE_YEAR = 2026
    CONDITION_ORDER = ["Poor", "Fair", "Good", "Excellent"]
    FEATURE_NUMERIC = ["sqft", "bedrooms", "bathrooms", "house_age"]
    FEATURE_ORDINAL = ["condition"]
    FEATURE_NOMINAL = ["location"]

    SRC_DIR = PIPELINE_DIR.parent
    DATA_DIR = SRC_DIR / "data"
    MODELS_DIR = SRC_DIR / "models"

    PREPROCESSED_TRAIN_PATH = DATA_DIR / "data_preprocessed_train.csv"
    FEATURED_TRAIN_PATH = DATA_DIR / "data_featured_train.csv"
    FEATURE_ENGINEER_PATH = MODELS_DIR / "feature_engineer.pkl"

    @classmethod
    def build_feature_engineer(cls) -> Pipeline:
        return Pipeline(
            [
                ("add_house_age", AddHouseAge(reference_year=cls.REFERENCE_YEAR)),
                (
                    "encode",
                    ColumnTransformer(
                        transformers=[
                            ("num", StandardScaler(), cls.FEATURE_NUMERIC),
                            (
                                "ord",
                                OrdinalEncoder(
                                    categories=[cls.CONDITION_ORDER],
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                                cls.FEATURE_ORDINAL,
                            ),
                            (
                                "nom",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                cls.FEATURE_NOMINAL,
                            ),
                        ],
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    ),
                ),
            ]
        )

    @classmethod
    def run(cls) -> None:
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(cls.PREPROCESSED_TRAIN_PATH)
        print(f"Rows: {train_df.shape[0]} | Columns: {train_df.shape[1]}")
        print(f"Columns: {list(train_df.columns)}")

        feature_engineer = cls.build_feature_engineer()
        feature_engineer.fit(train_df)
        train_featured = feature_engineer.transform(train_df)

        print(f"Rows: {train_featured.shape[0]} | Columns: {train_featured.shape[1]}")
        print(f"Columns: {list(train_featured.columns)}")

        train_featured.to_csv(cls.FEATURED_TRAIN_PATH, index=False)
        with open(cls.FEATURE_ENGINEER_PATH, "wb") as file:
            pickle.dump(feature_engineer, file)
        print(f"Wrote {cls.FEATURED_TRAIN_PATH}")
        print(f"Wrote {cls.FEATURE_ENGINEER_PATH}")


if __name__ == "__main__":
    DataPreprocessedTrainFeatureEngineering.run()
