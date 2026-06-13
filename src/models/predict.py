import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import config
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class DiscountPredictor:
    def __init__(
        self,
        discount_model_path: Optional[str] = None,
        slow_model_path: Optional[str] = None,
        feature_columns_path: Optional[str] = None,
    ):
        self.discount_model_path = discount_model_path or str(
            config.MODEL_DISCOUNT_FILE
        )
        self.slow_model_path = slow_model_path or str(config.MODEL_SLOW_RISK_FILE)
        self.feature_columns_path = feature_columns_path or str(
            config.FEATURE_COLUMNS_FILE
        )
        self.discount_model: Any = None
        self.slow_model: Any = None
        self.feature_columns: List[str] = []
        self.load_models()

    def load_models(self) -> None:
        logger.info("Loading discount model from %s", self.discount_model_path)
        self.discount_model = joblib.load(self.discount_model_path)
        logger.info(
            "Discount model loaded: %s", type(self.discount_model).__name__
        )

        logger.info("Loading slow-risk model from %s", self.slow_model_path)
        self.slow_model = joblib.load(self.slow_model_path)
        logger.info("Slow-risk model loaded: %s", type(self.slow_model).__name__)

        try:
            self.feature_columns = joblib.load(self.feature_columns_path)
            logger.info(
                "Feature columns loaded: %s", self.feature_columns
            )
        except FileNotFoundError:
            self.feature_columns = config.REQUIRED_FEATURES
            logger.warning(
                "Feature columns file not found, using defaults: %s",
                self.feature_columns,
            )

    def predict_discount(
        self, features_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.discount_model.predict(features_df)

        if hasattr(self.discount_model, "estimators_"):
            tree_preds = np.array(
                [
                    tree.predict(features_df)
                    for tree in self.discount_model.estimators_
                ]
            )
            std = np.std(tree_preds, axis=0)
            mean_pred = np.mean(tree_preds, axis=0)
            cv = std / (np.abs(mean_pred) + 1e-6)
            confidence = 1 / (1 + cv)
            confidence = np.clip(confidence, 0.1, 0.99)
        else:
            confidence = np.full(len(preds), 0.8)

        return preds, confidence

    def predict_slow_risk(self, features_df: pd.DataFrame) -> np.ndarray:
        probs = self.slow_model.predict_proba(features_df)[:, 1]
        return probs

    def predict(
        self, feature_rows: List[Dict]
    ) -> List[Dict]:
        features_df = pd.DataFrame(feature_rows)
        features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)

        discounts, confidences = self.predict_discount(features_df)
        slow_risks = self.predict_slow_risk(features_df)

        results = []
        for disc, conf, risk in zip(discounts, confidences, slow_risks):
            results.append(
                {
                    "recommended_discount": float(disc),
                    "confidence": float(conf),
                    "predicted_sales_lift": float(1.2 + 0.5 * disc),
                    "revenue_impact": float(disc * 100),
                    "slow_risk_probability": float(risk),
                }
            )
        return results

    def reload(self) -> None:
        logger.info("Reloading models...")
        self.load_models()
