from flask import Flask, jsonify, request
from flask_cors import CORS

import config
from src.models.predict import DiscountPredictor
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)
CORS(app)

predictor = DiscountPredictor()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ML service is running"}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        products = data.get("products", [])
        if not products:
            return jsonify({"error": "No products provided"}), 400

        product_ids = []
        feature_rows = []
        for prod in products:
            pid = prod.get("product_id")
            features = prod.get("features", {})
            product_ids.append(pid)
            feature_rows.append(features)

        predictions = predictor.predict(feature_rows)
        response_predictions = [
            {
                "product_id": pid,
                **pred,
            }
            for pid, pred in zip(product_ids, predictions)
        ]

        return jsonify({"predictions": response_predictions}), 200
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/reload", methods=["POST"])
def reload():
    predictor.reload()
    return jsonify({"status": "Models reloaded successfully"}), 200


if __name__ == "__main__":
    logger.info("Starting Flask server on %s:%s", config.FLASK_HOST, config.FLASK_PORT)
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False)
