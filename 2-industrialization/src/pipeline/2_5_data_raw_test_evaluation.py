"""2.5 Test evaluation.

Load raw test and apply saved preprocessor, feature engineer, and model
(transform / predict only).
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from custom_transformers import AddHouseAge  # noqa: E402, F401

set_config(transform_output="pandas")


class DataRawTestEvaluation:
    TARGET = "price"

    SRC_DIR = PIPELINE_DIR.parent
    DATA_DIR = SRC_DIR / "data"
    MODELS_DIR = SRC_DIR / "models"

    TEST_PATH = DATA_DIR / "data_raw_test.csv"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
    FEATURE_ENGINEER_PATH = MODELS_DIR / "feature_engineer.pkl"
    MODEL_PATH = MODELS_DIR / "model.pkl"
    PREDICTIONS_PATH = DATA_DIR / "data_featured_test_predictions.csv"

    @classmethod
    def run(cls) -> None:
        test_df = pd.read_csv(cls.TEST_PATH)
        print(f"Rows: {test_df.shape[0]} | Columns: {test_df.shape[1]}")
        print(f"Columns: {list(test_df.columns)}")

        with open(cls.PREPROCESSOR_PATH, "rb") as file:
            preprocessor = pickle.load(file)
        with open(cls.FEATURE_ENGINEER_PATH, "rb") as file:
            feature_engineer = pickle.load(file)
        with open(cls.MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        test_preprocessed = preprocessor.transform(test_df)
        test_featured = feature_engineer.transform(test_preprocessed)

        X_test = test_featured.drop(columns=[cls.TARGET])
        y_test = test_featured[cls.TARGET]
        X_test = X_test[list(model.feature_names_in_)]
        y_pred = model.predict(X_test)

        print(f"Test rows after transform: {len(X_test)}")
        print(f"Model: {type(model).__name__}")

        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
        test_r2 = r2_score(y_test, y_pred)
        print(f"Test R2:  {test_r2:.6f}")
        print(f"Test MAE: {test_mae:,.2f}")
        print(f"Test RMSE: {test_rmse:,.2f}")

        predictions = test_featured.copy()
        predictions["price_pred"] = y_pred
        predictions["error"] = predictions["price_pred"] - predictions[cls.TARGET]
        predictions["abs_error"] = predictions["error"].abs()
        predictions.to_csv(cls.PREDICTIONS_PATH, index=False)
        print(f"Wrote {cls.PREDICTIONS_PATH}")


if __name__ == "__main__":
    DataRawTestEvaluation.run()
