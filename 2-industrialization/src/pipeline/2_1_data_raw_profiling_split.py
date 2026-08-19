"""2.1 Raw data profiling and split.

Load data_raw.csv, print a structural profile, then write an 80/20 train/test split.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class DataRawProfilingSplit:
    RANDOM_STATE = 42
    TEST_SIZE = 0.20

    SRC_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = SRC_DIR / "data"

    RAW_PATH = DATA_DIR / "data_raw.csv"
    TRAIN_PATH = DATA_DIR / "data_raw_train.csv"
    TEST_PATH = DATA_DIR / "data_raw_test.csv"

    @classmethod
    def run(cls) -> None:
        df = pd.read_csv(cls.RAW_PATH)
        print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        print(f"Columns: {list(df.columns)}")

        df.info()

        schema = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "non_null": df.notna().sum(),
                "nulls": df.isna().sum(),
                "null_pct": (df.isna().mean() * 100).round(2),
                "n_unique": df.nunique(dropna=False),
            }
        )
        print(schema.to_string())
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print(f"Missing values: {int(df.isna().sum().sum())}")

        train_df, test_df = train_test_split(
            df,
            test_size=cls.TEST_SIZE,
            random_state=cls.RANDOM_STATE,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print(f"Train: {train_df.shape[0]} rows ({train_df.shape[0] / len(df):.0%})")
        print(f"Test : {test_df.shape[0]} rows ({test_df.shape[0] / len(df):.0%})")

        train_df.to_csv(cls.TRAIN_PATH, index=False)
        test_df.to_csv(cls.TEST_PATH, index=False)
        print(f"Wrote {cls.TRAIN_PATH}")
        print(f"Wrote {cls.TEST_PATH}")


if __name__ == "__main__":
    DataRawProfilingSplit.run()
