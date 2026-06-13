import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

MODELS_DIR = BASE_DIR / "models"

RAW_DATA_FILE = RAW_DATA_DIR / "superstore.xls"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "superstore_cleaned.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "X_features.csv"
TARGET_FILE = PROCESSED_DATA_DIR / "y_target.csv"
FEATURE_COLUMNS_FILE = PROCESSED_DATA_DIR / "feature_columns.pkl"

MODEL_DISCOUNT_FILE = MODELS_DIR / "discount_rf_v1.pkl"
MODEL_SLOW_RISK_FILE = MODELS_DIR / "slow_gb_v1.pkl"

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", 100))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", 10))
RF_RANDOM_STATE = 42

GB_N_ESTIMATORS = int(os.getenv("GB_N_ESTIMATORS", 50))
GB_RANDOM_STATE = 42

TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
SLOW_RISK_THRESHOLD = float(os.getenv("SLOW_RISK_THRESHOLD", 0.2))

REQUIRED_FEATURES = [
    "sales_velocity_7d",
    "sales_velocity_30d",
    "days_since_last_sale",
    "profit_margin",
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "current_stock",
]

API_REQUIRED_FIELDS = REQUIRED_FEATURES

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
