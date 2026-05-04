"""
Standalone Prometheus Metrics Exporter (Task 6)
- Exports data-level and model-level metrics on port 8000
- Prometheus scrapes this endpoint
"""
from prometheus_client import start_http_server, Gauge
import pandas as pd
import json
import time
import os

#DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
#INTERIM_DIR = os.path.join(DATA_DIR, "interim")

#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#DATA_DIR = os.path.join(BASE_DIR, "Data")
#INTERIM_DIR = os.path.join(DATA_DIR, "interim")
DATA_DIR = "/app/Data"
INTERIM_DIR = "/app/Data/interim"

# Data-level metrics
MISSING_VALUES = Gauge("data_missing_values_count", "Number of missing values in raw data")
FEATURE_MEAN_TXN_AMT = Gauge("feature_mean_transaction_amt", "Mean TransactionAmt")
FEATURE_STD_TXN_AMT = Gauge("feature_std_transaction_amt", "Std TransactionAmt")

# Model-level metrics
FRAUD_RECALL = Gauge("model_fraud_recall", "Current recall for fraud class")
FRAUD_PRECISION = Gauge("model_fraud_precision", "Current precision for fraud class")
CURRENT_AUC = Gauge("model_current_auc", "Current AUC-ROC")


def export_metrics():
    print("--- Starting Prometheus Metrics Exporter (Task 6) ---")
    start_http_server(8000)

    while True:
        # Data-level monitoring
        try:
            raw_path = os.path.join(INTERIM_DIR, "01_raw.csv")
            if os.path.exists(raw_path):
                df = pd.read_csv(raw_path, nrows=1000)
                MISSING_VALUES.set(int(df.isnull().sum().sum()))
                if "TransactionAmt" in df.columns:
                    FEATURE_MEAN_TXN_AMT.set(float(df["TransactionAmt"].mean()))
                    FEATURE_STD_TXN_AMT.set(float(df["TransactionAmt"].std()))
        except Exception:
            pass

        # Model-level monitoring from eval results
        try:
            details_path = os.path.join(INTERIM_DIR, "eval_details.json")
            if os.path.exists(details_path):
                with open(details_path, "r") as f:
                    details = json.load(f)
                best = details.get("2_XGBoost_Cost_Sensitive", {})
                FRAUD_RECALL.set(best.get("recall", 0))
                FRAUD_PRECISION.set(best.get("precision", 0))
                CURRENT_AUC.set(best.get("auc", 0))
        except Exception:
            pass

        time.sleep(15)


if __name__ == "__main__":
    export_metrics()
