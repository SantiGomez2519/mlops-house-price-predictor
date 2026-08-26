"""2.5 Test evaluation.

Load raw test and apply saved preprocessor, feature engineer, and model
(transform / predict only).
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from inference.predictor import HousePricePredictor

TARGET = "price"


class DataRawTestEvaluation:
    SRC_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = SRC_DIR / "data"
    MODELS_DIR = SRC_DIR / "models"

    TEST_PATH = DATA_DIR / "data_raw_test.csv"
    PREDICTIONS_PATH = DATA_DIR / "data_featured_test_predictions.csv"

    @classmethod
    def run(cls) -> None:
        test_df = pd.read_csv(cls.TEST_PATH)
        print(f"Rows: {test_df.shape[0]} | Columns: {test_df.shape[1]}")
        print(f"Columns: {list(test_df.columns)}")

        predictor = HousePricePredictor(cls.MODELS_DIR)
        result = predictor.predict_batch(test_df)

        y_test = result[TARGET]
        y_pred = result["price_pred"]

        print(f"Test rows: {len(result)}")
        print(f"Model: {type(predictor.model).__name__}")

        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
        test_r2 = r2_score(y_test, y_pred)
        print(f"Test R2:  {test_r2:.6f}")
        print(f"Test MAE: {test_mae:,.2f}")
        print(f"Test RMSE: {test_rmse:,.2f}")

        predictions = result.copy()
        predictions["error"] = predictions["price_pred"] - predictions[TARGET]
        predictions["abs_error"] = predictions["error"].abs()
        predictions.to_csv(cls.PREDICTIONS_PATH, index=False)
        print(f"Wrote {cls.PREDICTIONS_PATH}")


if __name__ == "__main__":
    DataRawTestEvaluation.run()
