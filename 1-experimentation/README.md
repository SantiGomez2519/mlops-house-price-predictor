# House price predictor — experimentation

Run the notebooks in order. Each step writes files that the next one reads.

## Setup

From this folder (`1-experimentation`):

```bash
uv sync
uv run jupyter notebook notebooks
```

Or open the `notebooks/` folder in VS Code / Cursor and select the same interpreter.

## Run order

| # | Notebook | What it does | Writes |
|---|---|---|---|
| 1 | `1_1_data_raw_profiling_split.ipynb` | Profile raw data and split 80/20 | `data/data_raw_train.csv`, `data/data_raw_test.csv` |
| 2 | `1_2_data_raw_train_preprocessing.ipynb` | Drop train duplicates and missing `price` (not in the pickle), then impute | `data/data_preprocessed_train.csv`, `models/preprocessor.pkl` |
| 3 | `1_3_data_preprocessed_train_exploratory_data_analysis.ipynb` | EDA on preprocessed train | — |
| 4 | `1_4_data_preprocessed_train_feature_engineering.ipynb` | `house_age`, scale numerics, ordinal `condition`, one-hot `location` | `data/data_featured_train.csv`, `models/feature_engineer.pkl` |
| 5 | `1_5_data_featured_train_training.ipynb` | Tune models with GridSearchCV, keep the best | `models/model.pkl`, `models/model_config.json` |
| 6 | `1_6_data_raw_test_evaluation.ipynb` | Apply saved artifacts to test (`transform`/`predict` only) | `data/data_featured_test_predictions.csv` |

Run every cell top to bottom in each notebook before opening the next one.

Starting input: `data/data_raw.csv`.
