# House price predictor — industrialization

Run the pipeline scripts in order. Each step writes files that the next one reads.

## Setup

From this folder (`2-industrialization`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

```bash
python src/pipeline/2_1_data_raw_profiling_split.py
python src/pipeline/2_2_data_raw_train_preprocessing.py
python src/pipeline/2_3_data_preprocessed_train_feature_engineering.py
python src/pipeline/2_4_data_raw_test_evaluation.py
```

| # | Script | What it does | Writes |
|---|---|---|---|
| 1 | `2_1_data_raw_profiling_split.py` | Profile raw data and split 80/20 | `src/data/data_raw_train.csv`, `src/data/data_raw_test.csv` |
| 2 | `2_2_data_raw_train_preprocessing.py` | Drop train duplicates and missing `price` (not in the pickle), then impute | `src/data/data_preprocessed_train.csv`, `src/models/preprocessor.pkl` |
| 3 | `2_3_data_preprocessed_train_feature_engineering.py` | `house_age`, scale numerics, ordinal `condition`, one-hot `location`, then tune models with GridSearchCV | `src/data/data_featured_train.csv`, `src/models/feature_engineer.pkl`, `src/models/model.pkl`, `src/models/model_config.json` |
| 4 | `2_4_data_raw_test_evaluation.py` | Apply saved artifacts to test (`transform`/`predict` only) | `src/data/data_featured_test_predictions.csv` |

Starting input: `src/data/data_raw.csv`.
