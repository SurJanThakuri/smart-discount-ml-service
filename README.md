# Dynamic Discount Recommender — ML Service

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

## Dataset

The Superstore Sales dataset (`data/raw/superstore.xls`) is included. It contains 1,000 sales records with columns: OrderID, OrderDate, ProductID, Sales, Profit, Discount, Quantity, and others.

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

The server starts on `http://localhost:5000` by default (configurable via `.env`).

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

### Reload Models

```
POST /api/reload
```

Reloads models from disk without restarting the server. Useful after retraining.

## Integrating with the Backend

The NestJS backend (`../server/`) runs independently and needs to call this ML service for predictions.

### Installation

In the server project, install the HTTP client:

```bash
cd ../server
npm install @nestjs/axios axios
```

### Environment Variable

Add to the server's `.env`:

```
ML_SERVICE_URL=http://localhost:5000
```

### Example NestJS Module

```typescript
// src/ml/ml.module.ts
import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { MlService } from './ml.service';
import { MlController } from './ml.controller';

@Module({
  imports: [HttpModule.register({
    baseURL: process.env.ML_SERVICE_URL || 'http://localhost:5000',
    timeout: 10000,
  })],
  controllers: [MlController],
  providers: [MlService],
  exports: [MlService],
})
export class MlModule {}
```

```typescript
// src/ml/ml.service.ts
import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class MlService {
  constructor(private readonly httpService: HttpService) {}

  async predictDiscounts(products: any[]) {
    const { data } = await firstValueFrom(
      this.httpService.post('/api/predict', { products })
    );
    return data.predictions;
  }

  async healthCheck() {
    const { data } = await firstValueFrom(
      this.httpService.get('/api/health')
    );
    return data;
  }
}
```

```typescript
// src/ml/ml.controller.ts
import { Controller, Post, Body } from '@nestjs/common';
import { MlService } from './ml.service';

@Controller('ml')
export class MlController {
  constructor(private readonly mlService: MlService) {}

  @Post('predict')
  async predict(@Body('products') products: any[]) {
    return this.mlService.predictDiscounts(products);
  }
}
```

### Expected Data Flow

```
Client App (React)  ──POST──>  NestJS Backend (:3003)
                                  │
                           POST /ml/predict
                                  │
                                  v
                         ML Service (:5000)
                           POST /api/predict
                                  │
                                  v
                        Returns discount predictions
```

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

Copy `.env.example` to `.env` and adjust as needed.

## Project Structure

```
ml-service/
├── app.py                    # Flask API entry point
├── config.py                 # Central configuration
├── run.py                    # CLI entry point (train/serve)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (user)
├── .env.example              # Environment variables (template)
├── README.md                 # This file
├── data/
│   ├── raw/
│   │   └── superstore.xls    # Original dataset
│   ├── processed/            # Cleaned & feature matrices
│   └── synthetic/            # Optional test data
├── models/                   # Trained .pkl files
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
- **Module import errors**: Run from the `ml-service/` root directory with the virtual environment activated.
- **`.xls` loading fails**: Ensure `xlrd` is installed (included in `requirements.txt`).
