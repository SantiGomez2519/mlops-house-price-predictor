# House price predictor — experimentation

Run the notebooks in order. Each step writes files that the next one reads.

## Setup

From this folder (`simple-version/1-experimentation`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks
```

Or open the `notebooks/` folder in VS Code / Cursor and select the same interpreter.

## Run order

| # | Notebook | What it does | Writes |
|---|---|---|---|
| 1 | `1_1_data_raw_profiling_split.ipynb` | Profile raw data and split 80/20 | `data/data_raw_train.csv`, `data/data_raw_test.csv` |
| 2 | `1_2_data_raw_train_preprocessing.ipynb` | Impute **train only** (median numeric, most frequent categorical) | `data/data_preprocessed_train.csv`, `models/preprocessor.pkl` |
| 3 | `1_3_data_preprocessed_train_exploratory_data_analysis.ipynb` | EDA on preprocessed train | — |
| 4 | `1_4_data_preprocessed_train_feature_engineering.ipynb` | `house_age`, ordinal `condition`, one-hot `location` | `data/data_featured_train.csv`, `models/feature_engineer.pkl` |
| 5 | `1_5_data_featured_train_training.ipynb` | Compare models with CV, keep the best | `models/model.pkl` |
| 6 | `1_6_data_raw_test_evaluation.ipynb` | Apply saved artifacts to test (`transform`/`predict` only) | `data/data_featured_test_predictions.csv` |

Run every cell top to bottom in each notebook before opening the next one.

Starting input: `data/data_raw.csv`.
