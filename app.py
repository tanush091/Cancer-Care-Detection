"""
Cancer Care Detection – Flask Web Application
Run: python app.py
Then open: http://127.0.0.1:5000
"""
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import logging
import sys

# ── Configure Logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Load saved model artefacts ───────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

def load_model_artifacts():
    """Load model, scaler, and feature names from disk with clear error messages."""
    required = ["cancer_model.pkl", "scaler.pkl", "feature_names.pkl"]
    for fname in required:
        fpath = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(fpath):
            logger.error(f"Missing model file: {fpath}")
            logger.info("Please run: python train_model.py first to generate the model files.")
            return None, None, None

    try:
        with open(os.path.join(MODEL_DIR, "cancer_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
            feature_names = pickle.load(f)
        logger.info("Model artifacts loaded successfully.")
        return model, scaler, feature_names
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return None, None, None

model, scaler, feature_names = load_model_artifacts()

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def home():
    if not model:
        return render_template("index.html", error="Model not loaded. Please run train_model.py first.", feature_names=[])
    return render_template("index.html", feature_names=feature_names)


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template("index.html", error="System Error: Model artifacts are missing.", feature_names=feature_names)
    try:
        num_features = len(feature_names)
        features = [float(request.form[f"feature{i}"]) for i in range(num_features)]
        arr = np.array(features).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        pred  = model.predict(arr_scaled)[0]
        proba = model.predict_proba(arr_scaled)[0]

        # Wisconsin dataset: target 0 = Malignant, 1 = Benign
        if pred == 0:
            result  = "Malignant"
            message = "⚠️ Tumour detected as Malignant (Cancerous). Please consult a doctor immediately."
            risk    = round(proba[0] * 100, 2)   # probability of class 0 (Malignant)
            badge   = "danger"
        else:
            result  = "Benign"
            message = "✅ Tumour detected as Benign (Non-Cancerous). Regular check-ups recommended."
            risk    = round(proba[1] * 100, 2)   # probability of class 1 (Benign)
            badge   = "safe"

        logger.info(f"Prediction made: {result} ({risk}%)")
        return render_template(
            "index.html",
            feature_names=feature_names,
            prediction=result,
            message=message,
            confidence=risk,
            badge=badge,
            user_values={f"feature{i}": request.form.get(f"feature{i}", "") for i in range(num_features)},
        )

    except KeyError as e:
        return render_template("index.html", feature_names=feature_names,
                               error=f"Missing input field: {e}. Please fill in all features.")
    except ValueError as e:
        return render_template("index.html", feature_names=feature_names,
                               error=f"Invalid number entered: {e}. All fields must be numeric.")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html", feature_names=feature_names, error="An internal error occurred.")


# ── REST API endpoint (optional / Postman / cURL) ───────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 503
    try:
        body     = request.get_json(force=True)
        num_features = len(feature_names)
        if "features" not in body:
            return jsonify({"error": "Request body must contain a 'features' list."}), 400
        features = body["features"]
        if len(features) != num_features:
            return jsonify({"error": f"Expected {num_features} features, got {len(features)}."}), 400
        arr   = np.array(features, dtype=float).reshape(1, -1)
        arr_s = scaler.transform(arr)
        pred  = model.predict(arr_s)[0]
        proba = model.predict_proba(arr_s)[0].tolist()
        return jsonify({
            "prediction":  "Malignant" if pred == 0 else "Benign",
            "probability": {"malignant": round(proba[0], 4), "benign": round(proba[1], 4)},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Cancer Care Detection App starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
