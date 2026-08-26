# House Price Predictor — MLOps Framework

End-to-end ML pipeline that predicts house prices from raw features. Organized in
four sequential phases, each in its own directory with independent dependencies.

## Architecture

```
shared/                         ← Shared package (inference + transformers)
│
1-experimentation/              ← Notebooks: prototyping and EDA
        ↓ data + pickles
2-industrialization/            ← Scripts: production pipeline
        ↓ pickles
3-serving/                      ← FastAPI: HTTP prediction service
        ↓ HTTP API
4-application/                  ← Vue.js: frontend SPA
```

### Data flow

```
data_raw.csv
    │
    ▼
[Profiling & Split]        →  data_raw_train.csv / data_raw_test.csv
    │
    ▼
[Preprocessing]            →  data_preprocessed_train.csv + preprocessor.pkl
    │
    ▼
[Feature Engineering]      →  data_featured_train.csv + feature_engineer.pkl
    │
    ▼
[Training]                 →  model.pkl + model_config.json
    │
    ▼
[Evaluation]               →  data_featured_test_predictions.csv
    │
    ▼
[Serving API]              →  POST /predict → { price_pred }
    │
    ▼
[Frontend]                 →  UI to enter features and view prediction
```

## Directory structure

```
mlops-house-price-predictor/
│
├── shared/                          ← Shared package
│   ├── pyproject.toml
│   ├── transformers/
│   │   └── custom_transformers.py   ← AddHouseAge (sklearn transformer)
│   └── inference/
│       └── predictor.py             ← HousePricePredictor (raw → predict)
│
├── 1-experimentation/               ← Phase 1: prototyping
│   ├── notebooks/                   ← 6 sequential Jupyter notebooks
│   ├── data/                        ← Input and output CSVs
│   └── models/                      ← Generated pickles
│
├── 2-industrialization/             ← Phase 2: production pipeline
│   └── src/
│       ├── pipeline/                ← 5 sequential CLI scripts
│       ├── data/                    ← CSVs (pipeline input/output)
│       └── models/                  ← Pickles (preprocessor, feature_engineer, model)
│
├── 3-serving/                       ← Phase 3: prediction API
│   └── src/
│       ├── app.py                   ← Application class (entry point)
│       ├── config.py                ← Settings (path resolution)
│       ├── schemas.py               ← Pydantic models (request/response)
│       ├── deps.py                  ← PredictorService (dependency injection)
│       ├── routers/
│       │   └── prediction.py        ← /health, /predict
│       └── models/
│
└── 4-application/                   ← Phase 4: frontend
    ├── src/App.vue                  ← SPA (form + prediction)
    └── vite.config.js              ← Proxy to API on :8000
```

## Tech stack

| Layer | Technology |
|---|---|
| Data | pandas |
| ML | scikit-learn 1.6.1 |
| Experimentation | Jupyter, matplotlib, seaborn |
| Packaging | uv (lockfiles) + setuptools |
| API | FastAPI + Pydantic |
| Frontend | Vue 3 + Vite 6 |

## Quick start

### 1. Experimentation (notebooks)

```bash
cd 1-experimentation
uv sync
uv run jupyter notebook notebooks
```

### 2. Industrialization (pipeline)

```bash
cd 2-industrialization
uv sync
uv run python src/pipeline/2_1_data_raw_profiling_split.py
uv run python src/pipeline/2_2_data_raw_train_preprocessing.py
uv run python src/pipeline/2_3_data_preprocessed_train_feature_engineering.py
uv run python src/pipeline/2_4_data_featured_train_training.py
uv run python src/pipeline/2_5_data_raw_test_evaluation.py
```

### 3. Serving (API)

```bash
cd 3-serving
uv sync
uv run fastapi dev src/app.py
```

API available at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

### 4. Frontend

```bash
cd 4-application
npm install
npm run dev
```

UI available at `http://127.0.0.1:5173`. Vite proxies `/predict` and `/health` to `:8000`.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Loaded model status |
| `POST` | `/predict` | Predict house price |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sqft": 1527, "bedrooms": 2, "bathrooms": 1.5, "location": "Suburb", "year_built": 1956, "condition": "Good"}'
```

## Model config

By default the serving loads pickles from `2-industrialization/src/models/`.
Override with the `MODELS_DIR` environment variable:

```bash
MODELS_DIR=/path/to/models uv run fastapi dev src/app.py
```

## Shared package

`shared/` is an installable Python package containing the inference logic
shared between industrialization and serving:

- `transformers/custom_transformers.py` — `AddHouseAge`: custom scikit-learn
  transformer that computes `house_age` from `year_built`.
- `inference/predictor.py` — `HousePricePredictor`: encapsulates the full chain
  `raw features → preprocessor → feature_engineer → model → prediction`.

Both consumers (`2-industrialization` and `3-serving`) import it as an editable
dependency via `[tool.uv.sources]` in their `pyproject.toml`.
