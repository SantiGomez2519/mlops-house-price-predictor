# House price predictor — application

Vue app that collects house features and calls the serving API to show a predicted
`price`.

## Setup

Start the API first (`3-serving`), then from this folder (`4-application`):

```bash
npm install
npm run dev
```

The UI starts at `http://127.0.0.1:5173`. Vite proxies `/predict` and `/health` to
`http://127.0.0.1:8000`.

## What it does

| Field | Sent to `POST /predict` |
|---|---|
| Square feet | `sqft` |
| Bedrooms | `bedrooms` |
| Bathrooms | `bathrooms` |
| Year built | `year_built` |
| Location | `location` |
| Condition | `condition` |

The predicted value is `price_pred` from the API.
