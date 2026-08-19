"""2.3 Feature engineering and training.

Fit house_age + scaling/encoding on preprocessed train, then tune models with
GridSearchCV and keep the best (experimentation 1.4 + 1.5).
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

set_config(transform_output="pandas")


class DataPreprocessedTrainFeatureEngineering:
    TARGET = "price"
    REFERENCE_YEAR = 2026
    CONDITION_ORDER = ["Poor", "Fair", "Good", "Excellent"]
    FEATURE_NUMERIC = ["sqft", "bedrooms", "bathrooms", "house_age"]
    FEATURE_ORDINAL = ["condition"]
    FEATURE_NOMINAL = ["location"]
    RANDOM_STATE = 42
    CV_SPLITS = 5
    BEST_SELECTION_METRIC = "mae"

    SRC_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = SRC_DIR / "data"
    MODELS_DIR = SRC_DIR / "models"

    PREPROCESSED_TRAIN_PATH = DATA_DIR / "data_preprocessed_train.csv"
    FEATURED_TRAIN_PATH = DATA_DIR / "data_featured_train.csv"
    FEATURE_ENGINEER_PATH = MODELS_DIR / "feature_engineer.pkl"
    MODEL_PATH = MODELS_DIR / "model.pkl"
    MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

    @staticmethod
    def add_house_age(X, reference_year=2026, source="year_built"):
        out = X.copy()
        out["house_age"] = reference_year - out[source]
        return out.drop(columns=[source])

    @classmethod
    def build_feature_engineer(cls) -> ColumnTransformer:
        return ColumnTransformer(
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
        )

    @classmethod
    def train_best_model(cls, X_train: pd.DataFrame, y_train: pd.Series):
        candidates = {
            "linear_regression": (LinearRegression(), {}),
            "ridge": (
                Ridge(),
                {"alpha": [0.1, 1.0, 10.0, 100.0]},
            ),
            "random_forest": (
                RandomForestRegressor(random_state=cls.RANDOM_STATE),
                {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5, None],
                    "min_samples_leaf": [1, 2],
                },
            ),
            "gradient_boosting": (
                GradientBoostingRegressor(random_state=cls.RANDOM_STATE),
                {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [2, 3],
                },
            ),
        }

        cv = KFold(n_splits=cls.CV_SPLITS, shuffle=True, random_state=cls.RANDOM_STATE)
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }

        searches = {}
        rows = []
        for name, (estimator, param_grid) in candidates.items():
            search = GridSearchCV(
                estimator,
                param_grid,
                cv=cv,
                scoring=scoring,
                refit=cls.BEST_SELECTION_METRIC,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            searches[name] = search
            best_idx = search.best_index_
            rows.append(
                {
                    "model": name,
                    "cv_r2": search.cv_results_["mean_test_r2"][best_idx],
                    "cv_mae": -search.cv_results_["mean_test_mae"][best_idx],
                    "cv_rmse": -search.cv_results_["mean_test_rmse"][best_idx],
                    "best_params": search.best_params_,
                }
            )

        comparison = pd.DataFrame(rows).sort_values(
            "cv_" + cls.BEST_SELECTION_METRIC
        ).reset_index(drop=True)
        print(comparison.to_string(index=False))

        best_name = comparison.loc[0, "model"]
        model = searches[best_name].best_estimator_
        print(f"Best model: {best_name}")
        print(f"Best params: {comparison.loc[0, 'best_params']}")
        return model, best_name, comparison

    @classmethod
    def run(cls) -> None:
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(cls.PREPROCESSED_TRAIN_PATH)
        print(f"Rows: {train_df.shape[0]} | Columns: {train_df.shape[1]}")
        print(f"Columns: {list(train_df.columns)}")

        train_df = cls.add_house_age(train_df, reference_year=cls.REFERENCE_YEAR)
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

        X_train = train_featured.drop(columns=[cls.TARGET])
        y_train = train_featured[cls.TARGET]
        print(f"Rows: {len(X_train)} | Features: {X_train.shape[1]}")

        model, best_name, comparison = cls.train_best_model(X_train, y_train)
        best_metrics = (
            comparison.loc[0, ["cv_r2", "cv_mae", "cv_rmse"]].astype(float).to_dict()
        )
        model_config = {
            "model_name": best_name,
            "model_class": type(model).__name__,
            "params": model.get_params(),
            "best_params": comparison.loc[0, "best_params"],
            "features": list(X_train.columns),
            "target": cls.TARGET,
            "cv": {
                "n_splits": cls.CV_SPLITS,
                "random_state": cls.RANDOM_STATE,
                "selection_metric": "cv_" + cls.BEST_SELECTION_METRIC,
                "metrics": best_metrics,
            },
        }

        with open(cls.MODEL_PATH, "wb") as file:
            pickle.dump(model, file)
        with open(cls.MODEL_CONFIG_PATH, "w", encoding="utf-8") as file:
            json.dump(model_config, file, indent=2, default=str)

        print(f"Wrote {cls.MODEL_PATH} ({best_name})")
        print(f"Wrote {cls.MODEL_CONFIG_PATH}")


if __name__ == "__main__":
    DataPreprocessedTrainFeatureEngineering.run()
