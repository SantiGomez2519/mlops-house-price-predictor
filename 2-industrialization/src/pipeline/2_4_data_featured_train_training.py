"""2.4 Training.

Tune models with GridSearchCV on featured train and keep the best
(experimentation 1.5).
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold


class DataFeaturedTrainTraining:
    TARGET = "price"
    RANDOM_STATE = 42
    CV_SPLITS = 5
    BEST_SELECTION_METRIC = "mae"

    SRC_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = SRC_DIR / "data"
    MODELS_DIR = SRC_DIR / "models"

    FEATURED_TRAIN_PATH = DATA_DIR / "data_featured_train.csv"
    MODEL_PATH = MODELS_DIR / "model.pkl"
    MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

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

        train_df = pd.read_csv(cls.FEATURED_TRAIN_PATH)
        X_train = train_df.drop(columns=[cls.TARGET])
        y_train = train_df[cls.TARGET]
        print(f"Rows: {len(X_train)} | Features: {X_train.shape[1]}")
        print(f"Columns: {list(X_train.columns)}")

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
    DataFeaturedTrainTraining.run()
