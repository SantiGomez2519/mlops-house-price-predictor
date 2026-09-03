"""2.2 Train preprocessing.

Drop duplicate rows and missing price in pandas (not stored in the pickle), then fit
an imputer ColumnTransformer on train only.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

set_config(transform_output="pandas")


class DataRawTrainPreprocessing:
    TARGET = "price"
    FEATURE_NUMERIC = ["sqft", "bedrooms", "bathrooms", "year_built"]
    FEATURE_CATEGORICAL = ["location", "condition"]

    SRC_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = SRC_DIR / "data"
    SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
    MODELS_DIR = SHARED_DIR / "models"

    TRAIN_PATH = DATA_DIR / "data_raw_train.csv"
    PREPROCESSED_TRAIN_PATH = DATA_DIR / "data_preprocessed_train.csv"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

    @classmethod
    def run(cls) -> None:
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(cls.TRAIN_PATH)
        print(
            f"Before: {train_df.shape[0]} rows | "
            f"duplicates={train_df.duplicated().sum()} | "
            f"missing price={int(train_df[cls.TARGET].isna().sum())}"
        )

        train_df = (
            train_df.drop_duplicates()
            .dropna(subset=[cls.TARGET])
            .reset_index(drop=True)
        )
        print(
            f"After:  {train_df.shape[0]} rows | "
            f"duplicates={train_df.duplicated().sum()} | "
            f"missing price={int(train_df[cls.TARGET].isna().sum())}"
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), cls.FEATURE_NUMERIC),
                ("cat", SimpleImputer(strategy="most_frequent"), cls.FEATURE_CATEGORICAL),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        preprocessor.fit(train_df)
        train_preprocessed = preprocessor.transform(train_df)[train_df.columns]

        print(f"After preprocessor: {train_preprocessed.shape[0]} rows")

        train_preprocessed.to_csv(cls.PREPROCESSED_TRAIN_PATH, index=False)
        with open(cls.PREPROCESSOR_PATH, "wb") as file:
            pickle.dump(preprocessor, file)

        print(f"Wrote {cls.PREPROCESSED_TRAIN_PATH}")
        print(f"Wrote {cls.PREPROCESSOR_PATH}")


if __name__ == "__main__":
    DataRawTrainPreprocessing.run()
