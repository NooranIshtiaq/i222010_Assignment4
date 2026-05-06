"""
Inference API — FastAPI (Task 6: Observability)
- Serves fraud predictions from MLflow champion model
- Exposes Prometheus metrics: system, model, and data-level
"""
import pandas as pd
import numpy as np
import mlflow.pyfunc
import time
import os
from collections import deque
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import requests
from fastapi import Request

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

GITHUB_USER = "NooranIshtiaq"
GITHUB_REPO = "i222010_Assignment4"


app = FastAPI(title="Fraud Detection API")

# --- SYSTEM METRICS (Task 6A) ---
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["http_status"])
LATENCY = Histogram("api_prediction_latency_seconds", "Inference Latency",
                    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
ERROR_COUNT = Counter("api_errors_total", "Total 5xx Errors")

# --- MODEL METRICS (Task 6B) ---
FRAUD_CONFIDENCE = Gauge("prediction_confidence_avg", "Avg Prediction Confidence")
MODEL_RECALL = Gauge("model_recall_score", "Current Recall Score")
MODEL_PRECISION = Gauge("model_precision_score", "Current Precision Score")
MODEL_FPR = Gauge("model_fpr_score", "False Positive Rate")
FRAUD_RATE = Gauge("prediction_fraud_rate", "Fraction of predictions that are fraud")

# --- DATA METRICS (Task 6C) ---
NULL_COUNT = Gauge("api_null_values_total", "Missing values in input")
FEATURE_DRIFT_SCORE = Gauge("feature_drift_score", "Feature distribution drift score")
INPUT_ANOMALY_COUNT = Counter("input_anomaly_total", "Input data anomalies detected")

model = None
# Track recent predictions for live metric computation
recent_predictions = deque(maxlen=500)
recent_actuals = deque(maxlen=500)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_champion_model(retries=10, delay=5):
    global model
    if model is None:
        for i in range(retries):
            try:
                mlflow.set_tracking_uri(MLFLOW_URI)
                # Check if model exists first to avoid confusing error logs
                model = mlflow.pyfunc.load_model(model_uri="models:/FraudModelV1/latest")
                print("Champion Model loaded successfully!")
                return
            except Exception as e:
                print(f"Model load attempt {i+1}/{retries} failed: {e}")
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    print("Max retries reached. Model load failed.")


@app.on_event("startup")
def startup_event():
    load_champion_model()
    MODEL_RECALL.set(0.0)
    MODEL_PRECISION.set(0.0)
    MODEL_FPR.set(0.0)
    NULL_COUNT.set(0)
    FEATURE_DRIFT_SCORE.set(0.0)


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(data: dict):
    if model is None:
        load_champion_model()
        if model is None:
            REQUEST_COUNT.labels(http_status="503").inc()
            return JSONResponse(status_code=503, content={"error": "Model not ready"})

    start_time = time.time()
    try:
        df = pd.DataFrame([data])

        # Data-level monitoring
        null_count = int(df.isnull().sum().sum())
        NULL_COUNT.set(null_count)
        if null_count > len(df.columns) * 0.3:
            INPUT_ANOMALY_COUNT.inc()

        # Feature alignment
        expected_features = None
        try:
            if model.metadata and model.metadata.signature:
                expected_features = model.metadata.signature.inputs.input_names()
        except Exception:
            pass

        if expected_features:
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected_features]
        else:
            # Fallback padding to 414 features if signature is missing
            current_cols = len(df.columns)
            if current_cols < 414:
                for i in range(414 - current_cols):
                    df[f"padded_feat_{i}"] = 0
            df = df.iloc[:, :414]

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        prediction = model.predict(df)
        pred_value = int(prediction[0])

        # Compute confidence (probability if available)
        try:
            confidence = float(max(prediction[0], 1 - prediction[0]))
        except Exception:
            confidence = 1.0

        # Track predictions for live metrics
        actual = data.get("isFraud")
        recent_predictions.append(pred_value)
        if actual is not None:
            recent_actuals.append(int(actual))

            # Compute live metrics if we have enough samples
            if len(recent_predictions) >= 10:
                preds = np.array(recent_predictions)
                acts = np.array(recent_actuals)

                tp = np.sum((preds == 1) & (acts == 1))
                fp = np.sum((preds == 1) & (acts == 0))
                tn = np.sum((preds == 0) & (acts == 0))
                fn = np.sum((preds == 0) & (acts == 1))

                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                MODEL_RECALL.set(recall)
                MODEL_PRECISION.set(precision)
                MODEL_FPR.set(fpr)

        fraud_rate = sum(recent_predictions) / len(recent_predictions)
        FRAUD_RATE.set(fraud_rate)
        FRAUD_CONFIDENCE.set(confidence)

        duration = time.time() - start_time
        LATENCY.observe(duration)
        REQUEST_COUNT.labels(http_status="200").inc()

        return {"is_fraud": pred_value, "confidence": round(confidence, 4),
                "latency": f"{duration:.4f}s"}

    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(http_status="500").inc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/webhook")
async def webhook(request: Request):
    alert = await request.json()
    print("🚨 Alert received:", alert)

    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/dispatches"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "event_type": "retrain_trigger"
    }

    response = requests.post(url, json=payload, headers=headers)

    return {
        "status": "trigger_sent",
        "github_status": response.status_code
    }

# Mount Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())
