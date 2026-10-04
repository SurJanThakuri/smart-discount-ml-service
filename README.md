# Smart Discount Recommender — ML Service

Flask-based ML microservice that recommends optimal discount percentages and identifies slow-moving stock risk using ensemble models (Random Forest + Gradient Boosting) trained on Superstore sales data.

## Requirements

- Python 3.12+
- Dependencies listed in `requirements.txt`

## Setup

```bash
cd ml-service

# Create virtual environment
python -m venv venv312

# Activate (Windows)
venv312\Scripts\activate
# Activate (Linux/Mac)
source venv312/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> The `src/` package's `__init__.py` files are required — all modules import each other as `src.data.loader`, `src.models.train`, etc. Always run commands from the `ml-service/` root.

## Where This Sits in the Project

```
React (:5173)
   │  POST /ml/predict            (NestJS proxy)
   ▼
NestJS (:3003)  ──►  Flask (:5000)
   │                   POST /api/predict
   │                   GET  /api/health
   │                   POST /api/reload
   ▼
PostgreSQL — rows written to `recommendations`
```

Two clients reach this service:

1. **NestJS** — `recommend.service.generate()` builds feature vectors from real sales data and persists the predictions. If this call fails, NestJS falls back to an in-process heuristic.
2. **React** — `useDiscountPredictions()` calls `/ml/predict` for live previews. If it fails, the client falls back to rules in `client/src/hooks/useDiscounts.ts`.

So the whole stack runs degraded-but-functional without this service.

## Dataset

The Superstore Sales dataset (`data/raw/superstore.xls`) is included. It contains 1,000 sales records with columns: OrderID, OrderDate, ShipDate, ShipMode, CustomerID, CustomerName, Segment, City, State, PostalCode, Region, ProductID, Category, Sub-Category, ProductName, Sales, Quantity, Discount, Profit.

To use your own data, place a `.xls` or `.csv` file in `data/raw/` and update `config.py`'s `RAW_DATA_FILE` path.

## Train Models

```bash
python run.py train
```

The pipeline:
1. Loads raw data from `data/raw/superstore.xls`
2. Auto-detects column names (case-insensitive)
3. Engineers features: profit margin, time features (day/week/month/quarter), sales velocity (7d & 30d), days since last sale
4. Splits data into train/test (80/20)
5. Trains a **RandomForestRegressor** (discount prediction) and a **GradientBoostingClassifier** (slow-risk classification)
6. Evaluates and logs MAE, R², and F1 scores
7. Saves models to `models/discount_rf_v1.pkl` and `models/slow_gb_v1.pkl`
8. Saves feature column order to `data/processed/feature_columns.pkl`

## Run the API

```bash
python run.py serve
```

Or directly:

```bash
python app.py
```

The server starts on `http://localhost:5000` by default (configurable via `.env`). Both trained models are loaded at startup.

## API Endpoints

### Health Check

```
GET /api/health
```

Response:
```json
{
  "status": "ML service is running"
}
```

### Predict Discounts

```
POST /api/predict
Content-Type: application/json
```

Request body:
```json
{
  "products": [
    {
      "product_id": "uuid-1",
      "features": {
        "sales_velocity_7d": 5.2,
        "sales_velocity_30d": 4.8,
        "days_since_last_sale": 12,
        "profit_margin": 0.32,
        "day_of_week": 3,
        "month": 6,
        "quarter": 2,
        "is_weekend": 0,
        "current_stock": 45
      }
    }
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "product_id": "uuid-1",
      "recommended_discount": 0.15,
      "confidence": 0.87,
      "predicted_sales_lift": 1.275,
      "revenue_impact": 15.0,
      "slow_risk_probability": 0.23
    }
  ]
}
```

**Feature Descriptions:**

| Feature | Type | Description |
|---|---|---|
| `sales_velocity_7d` | float | Average units sold per day over last 7 days |
| `sales_velocity_30d` | float | Average units sold per day over last 30 days |
| `days_since_last_sale` | int | Days since this product was last sold |
| `profit_margin` | float (0-1) | Profit / Sales ratio, clipped to [0, 1] |
| `day_of_week` | int (0-6) | Monday=0, Sunday=6 |
| `month` | int (1-12) | Calendar month |
| `quarter` | int (1-4) | Fiscal quarter |
| `is_weekend` | int (0/1) | 1 if Saturday or Sunday |
| `current_stock` | int | Current inventory level |

**Response Fields:**

| Field | Description |
|---|---|
| `recommended_discount` | Predicted optimal discount (0-1 range) |
| `confidence` | Model confidence score (0-1), derived from tree variance |
| `predicted_sales_lift` | Estimated sales multiplier after discount |
| `revenue_impact` | Estimated revenue change percentage |
| `slow_risk_probability` | Probability this product is at risk of being slow-moving (0-1) |

### Batch Predict

```
POST /api/predict/batch
```

Same payload/response shape as `/api/predict`, processed in chunks for large catalogs.

### Retrain

```
POST /api/retrain
```

Re-runs the full training pipeline on the configured dataset and writes fresh `.pkl` files.

### Reload Models

```
POST /api/reload
```

Reloads models from disk without restarting the server. Useful after retraining — this is also exposed by the NestJS proxy as `POST /ml/reload`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_HOST` | `0.0.0.0` | Flask binding address |
| `FLASK_PORT` | `5000` | Flask port |
| `RF_N_ESTIMATORS` | `100` | Random Forest number of trees |
| `RF_MAX_DEPTH` | `10` | Random Forest max tree depth |
| `GB_N_ESTIMATORS` | `50` | Gradient Boosting estimators |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `SLOW_RISK_THRESHOLD` | `0.2` | Discount threshold for slow-risk classification |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |

All values are read by `config.py` via `python-dotenv`, with the defaults shown above — a `.env` file is optional and only needs to override what you want to change.

## Project Structure

```
ml-service/
├── app.py                    # Flask API entry point (routes)
├── config.py                 # Paths + env configuration
├── run.py                    # CLI entry point (train/serve)
├── requirements.txt          # Python dependencies
├── .env                      # Environment overrides (optional)
├── README.md                 # This file
├── venv312/                  # Virtual environment
├── data/
│   ├── raw/
│   │   └── superstore.xls    # Original dataset
│   ├── processed/            # Cleaned data, features, feature_columns.pkl
│   └── synthetic/            # Optional test data
├── models/                   # discount_rf_v1.pkl, slow_gb_v1.pkl
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py         # Load raw/cleaned data
    │   └── preprocess.py     # Data cleaning
    ├── features/
    │   ├── __init__.py
    │   └── build_features.py # Feature engineering
    ├── models/
    │   ├── __init__.py
    │   ├── train.py          # Training pipeline
    │   └── predict.py        # Inference logic
    └── utils/
        ├── __init__.py
        └── helpers.py        # Logging utilities
```

## Troubleshooting

- **Missing columns error**: Ensure `data/raw/superstore.xls` contains the expected columns. The loader auto-detects by case-insensitive matching.
- **Model not found**: Run `python run.py train` first to generate model files.
- **Port already in use**: Change `FLASK_PORT` in `.env`.
- **Module import errors**: Run from the `ml-service/` root directory with the virtual environment activated — `src/` subpackages must keep their `__init__.py` files.
- **`.xls` loading fails**: Ensure `xlrd` is installed (included in `requirements.txt`).
- **NestJS can't reach it**: Check `ML_SERVICE_URL` in `server/.env` (default `http://localhost:5000`) and confirm `GET /api/health` responds.
