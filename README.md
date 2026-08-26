# House Price Predictor — MLOps Framework

End-to-end ML pipeline that predicts house prices from raw features. Organized in
four sequential phases, each in its own directory with independent dependencies.

## Architecture

```
shared/                         ← Paquete compartido (inference + transformers)
│
1-experimentation/              ← Notebooks: prototipado y EDA
        ↓ data + pickles
2-industrialization/            ← Scripts: pipeline de producción
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
[Frontend]                 →  UI para ingresar features y ver predicción
```

## Directory structure

```
mlops-house-price-predictor/
│
├── shared/                          ← Paquete compartido
│   ├── pyproject.toml
│   ├── transformers/
│   │   └── custom_transformers.py   ← AddHouseAge (sklearn transformer)
│   └── inference/
│       └── predictor.py             ← HousePricePredictor (raw → predict)
│
├── 1-experimentation/               ← Fase 1: prototipado
│   ├── notebooks/                   ← 6 Jupyter notebooks secuenciales
│   ├── data/                        ← CSVs de entrada y salida
│   └── models/                      ← Pickles generados
│
├── 2-industrialization/             ← Fase 2: pipeline de producción
│   └── src/
│       ├── pipeline/                ← 5 scripts CLI secuenciales
│       ├── data/                    ← CSVs (input/output del pipeline)
│       └── models/                  ← Pickles (preprocessor, feature_engineer, model)
│
├── 3-serving/                       ← Fase 3: API de predicción
│   └── src/
│       ├── app.py                   ← Application class (entry point)
│       ├── config.py                ← Settings (resolución de paths)
│       ├── schemas.py               ← Pydantic models (request/response)
│       ├── deps.py                  ← PredictorService (dependency injection)
│       ├── routers/
│       │   └── prediction.py        ← /health, /predict
│       └── models/
│
└── 4-application/                   ← Fase 4: frontend
    ├── src/App.vue                  ← SPA (form + predicción)
    └── vite.config.js              ← Proxy a la API en :8000
```

## Tech stack

| Capa | Tecnología |
|---|---|
| Datos | pandas |
| ML | scikit-learn 1.6.1 |
| Experimentación | Jupyter, matplotlib, seaborn |
| Empaquetado | uv (lockfiles) + setuptools |
| API | FastAPI + Pydantic |
| Frontend | Vue 3 + Vite 6 |

## Quick start

### 1. Experimentación (notebooks)

```bash
cd 1-experimentation
uv sync
uv run jupyter notebook notebooks
```

### 2. Industrialización (pipeline)

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

API disponible en `http://127.0.0.1:8000`. Documentación interactiva en `/docs`.

### 4. Frontend

```bash
cd 4-application
npm install
npm run dev
```

UI disponible en `http://127.0.0.1:5173`. Vite proxea `/predict` y `/health` a `:8000`.

## API endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del modelo cargado |
| `POST` | `/predict` | Predecir precio de una casa |

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sqft": 1527, "bedrooms": 2, "bathrooms": 1.5, "location": "Suburb", "year_built": 1956, "condition": "Good"}'
```

## Model config

Por defecto el serving carga los pickles de `2-industrialization/src/models/`.
Se puede sobreescricir con la variable de entorno `MODELS_DIR`:

```bash
MODELS_DIR=/path/to/models uv run fastapi dev src/app.py
```

## Shared package

`shared/` es un paquete Python instalable que contiene la lógica de inferencia
compartida entre industrialización y serving:

- `transformers/custom_transformers.py` — `AddHouseAge`: transformer custom de
  scikit-learn que calcula `house_age` a partir de `year_built`.
- `inference/predictor.py` — `HousePricePredictor`: encapsula la cadena completa
  `raw features → preprocessor → feature_engineer → model → prediction`.

Ambos consumeres (`2-industrialization` y `3-serving`) lo importan como
dependencia editable via `[tool.uv.sources]` en su `pyproject.toml`.
