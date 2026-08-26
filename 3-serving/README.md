# House price predictor — serving

HTTP API that loads the pickled preprocessor, feature engineer, and model, then
returns a predicted `price`.

## Setup

Train artifacts first (`2-industrialization`), then from this folder (`3-serving`):

```bash
uv sync
uv run fastapi dev src/app.py
```

The API starts at `http://127.0.0.1:8000`. Docs: `http://127.0.0.1:8000/docs`.

By default it reads pickles from `2-industrialization/src/models/`. Override with:

```bash
MODELS_DIR=/path/to/models uv run fastapi dev src/app.py
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Check that artifacts loaded |
| `POST` | `/predict` | Predict `price` for one house (`transform` / `predict` only) |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sqft": 1527, "bedrooms": 2, "bathrooms": 1.5, "location": "Suburb", "year_built": 1956, "condition": "Good"}'
```
