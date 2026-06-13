import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config
from src.data.loader import load_raw_data, detect_column_mapping
from src.data.preprocess import clean_data, get_column_map
from src.features.build_features import build_features
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def train_pipeline() -> None:
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    df = load_raw_data(config.RAW_DATA_FILE)
    df = clean_data(df)
    col_map = get_column_map(df)
    discount_col = col_map.get("discount", "Discount")
    y_discount_raw = df[discount_col].values
    X = build_features(df)

    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Target shape: %s", y_discount_raw.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_discount_raw, test_size=config.TEST_SIZE, random_state=42
    )

    logger.info("Training RandomForestRegressor...")
    rf = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RF_RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(
        "Discount Predictor - MAE: %.4f, R²: %.4f", mae, r2
    )

    logger.info("Training GradientBoostingClassifier (slow-risk)...")
    y_slow = (y_discount_raw > config.SLOW_RISK_THRESHOLD).astype(int)
    gb = GradientBoostingClassifier(
        n_estimators=config.GB_N_ESTIMATORS,
        random_state=config.GB_RANDOM_STATE,
    )
    gb.fit(X_train, y_slow[: len(X_train)])
    y_pred_slow = gb.predict(X_test)
    f1 = f1_score(y_slow[: len(X_test)], y_pred_slow)
    logger.info("Slow Stock Classifier - F1: %.4f", f1)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, config.MODEL_DISCOUNT_FILE)
    joblib.dump(gb, config.MODEL_SLOW_RISK_FILE)

    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, config.FEATURE_COLUMNS_FILE)

    logger.info("Models saved to %s", config.MODELS_DIR)
    logger.info("Feature columns saved to %s", config.FEATURE_COLUMNS_FILE)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    train_pipeline()
