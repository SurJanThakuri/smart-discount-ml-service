import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow requests from NestJS backend

# Load models at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models")
discount_model = joblib.load(f"{MODEL_PATH}/discount_rf.pkl")
slow_model = joblib.load(f"{MODEL_PATH}/slow_gb.pkl")
# elasticity_model can be added later

def predict_discount(features_df):
    """Return predicted discount % and confidence (simple std dev from trees)."""
    preds = discount_model.predict(features_df)
    # For Random Forest, estimate confidence as 1 - (std of tree predictions / mean)
    if hasattr(discount_model, 'estimators_'):
        tree_preds = np.array([tree.predict(features_df) for tree in discount_model.estimators_])
        std = np.std(tree_preds, axis=0)
        mean_pred = np.mean(tree_preds, axis=0)
        confidence = 1 - (std / (mean_pred + 1e-6))
        confidence = np.clip(confidence, 0, 1)
    else:
        confidence = [0.8] * len(preds)
    return preds, confidence

def predict_slow_risk(features_df):
    """Return probability of becoming dead stock."""
    probs = slow_model.predict_proba(features_df)[:, 1]
    return probs

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ML service is running"}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Expected JSON payload:
    {
      "products": [
        {
          "product_id": "uuid1",
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
    """
    data = request.get_json()
    products = data.get("products", [])
    if not products:
        return jsonify({"error": "No products provided"}), 400
    
    # Build feature matrix
    feature_rows = []
    product_ids = []
    for prod in products:
        product_ids.append(prod["product_id"])
        f = prod["features"]
        required = ['sales_velocity_7d', 'sales_velocity_30d', 'days_since_last_sale',
                    'profit_margin', 'day_of_week', 'month', 'quarter', 'is_weekend', 'current_stock']
        row = [f.get(k, 0) for k in required]
        feature_rows.append(row)
    
    feature_names = required
    features_df = pd.DataFrame(feature_rows, columns=feature_names)
    
    # Get predictions
    discounts, confidences = predict_discount(features_df)
    slow_risks = predict_slow_risk(features_df)
    
    # Placeholder revenue impact
    revenue_impacts = [d * 100 for d in discounts]
    
    response = {
        "predictions": [
            {
                "product_id": pid,
                "recommended_discount": float(disc),
                "confidence": float(conf),
                "predicted_sales_lift": float(1.2 + 0.5 * disc),
                "revenue_impact": float(rev),
                "slow_risk_probability": float(risk)
            }
            for pid, disc, conf, rev, risk in zip(product_ids, discounts, confidences, revenue_impacts, slow_risks)
        ]
    }
    return jsonify(response), 200

if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)