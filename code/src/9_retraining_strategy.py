"""
Step 9: Intelligent Retraining Strategy (Task 8)
- Hybrid strategy: threshold-based + periodic
- If recall < threshold → immediate retraining
- If last retrain > 7 days → periodic retraining
- Compares stability, cost, and performance improvement
"""
import mlflow
import pandas as pd
import numpy as np
import json
import time
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score

#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#DATA_DIR = os.path.join(BASE_DIR, "Data")
#INTERIM_DIR = os.path.join(DATA_DIR, "interim")
DATA_DIR = "/app/Data"
INTERIM_DIR = "/app/Data/interim"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "plots")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

RECALL_THRESHOLD = 0.85
PERIODIC_INTERVAL_DAYS = 7


def check_retraining_needed(drift_report, last_retrain_file):
    """Hybrid strategy: threshold + periodic."""
    recall = drift_report.get("drifted_recall", 1.0)
    drift_detected = drift_report.get("drift_detected", False)

    # Strategy 1: Threshold-based
    if recall < RECALL_THRESHOLD:
        return "threshold", f"Recall ({recall:.4f}) < threshold ({RECALL_THRESHOLD})"

    # Strategy 2: Drift-triggered
    if drift_detected:
        return "drift", "Data drift detected in previous step"

    # Strategy 3: Periodic
    try:
        with open(last_retrain_file, "r") as f:
            last_retrain = json.load(f).get("timestamp", 0)
        days_since = (time.time() - last_retrain) / 86400
        if days_since > PERIODIC_INTERVAL_DAYS:
            return "periodic", f"{days_since:.1f} days since last retrain (> {PERIODIC_INTERVAL_DAYS})"
    except FileNotFoundError:
        return "periodic", "No previous retraining record found"

    return None, "No retraining needed"


def retrain_model():
    """Retrain the model on the latest data."""
    df = pd.read_csv(os.path.join(INTERIM_DIR, "04_engineered.csv"))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df.drop(["isFraud", "TransactionID"], axis=1, errors="ignore")
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    start_time = time.time()
    model = XGBClassifier(scale_pos_weight=10, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    return model, recall, auc, train_time


def run_retraining_strategy():
    print("--- Step 9: Intelligent Retraining Strategy (Task 8) ---")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    last_retrain_file = os.path.join(INTERIM_DIR, "last_retrain.json")

    # Load drift report
    drift_path = os.path.join(INTERIM_DIR, "drift_report.json")
    try:
        with open(drift_path, "r") as f:
            drift_report = json.load(f)
    except FileNotFoundError:
        drift_report = {}

    # Load current model metrics
    eval_path = os.path.join(INTERIM_DIR, "best_auc.json")
    try:
        with open(eval_path, "r") as f:
            current_metrics = json.load(f)
            current_auc = current_metrics.get("best_auc", 0)
    except FileNotFoundError:
        current_auc = 0

    # ---- Check if retraining is needed ----
    trigger, reason = check_retraining_needed(drift_report, last_retrain_file)

    print(f"  Strategy Decision: {trigger or 'skip'}")
    print(f"  Reason: {reason}")

    strategy_results = []

    if trigger:
        print(f"\n  → Retraining triggered ({trigger})...")
        model, new_recall, new_auc, train_time = retrain_model()

        improvement = new_auc - current_auc
        print(f"  Previous AUC: {current_auc:.4f}")
        print(f"  New AUC:      {new_auc:.4f} (Δ {improvement:+.4f})")
        print(f"  New Recall:   {new_recall:.4f}")
        print(f"  Training time: {train_time:.1f}s")

        # Update last retrain timestamp
        with open(last_retrain_file, "w") as f:
            json.dump({"timestamp": time.time(), "trigger": trigger, "reason": reason}, f)

        # ---- Strategy Comparison Report ----
        strategy_results = [
            {"Strategy": "Threshold-based", "Trigger": f"Recall < {RECALL_THRESHOLD}",
             "Stability": "High — retrains only when needed",
             "Compute_Cost": "Low — on-demand",
             "Performance": f"AUC improvement: {improvement:+.4f}"},
            {"Strategy": "Periodic", "Trigger": f"Every {PERIODIC_INTERVAL_DAYS} days",
             "Stability": "Medium — may retrain unnecessarily",
             "Compute_Cost": "Medium — fixed schedule",
             "Performance": "Consistent but may waste resources"},
            {"Strategy": "Hybrid (Used)", "Trigger": "Threshold OR periodic OR drift",
             "Stability": "Highest — covers all scenarios",
             "Compute_Cost": f"{train_time:.1f}s per retrain",
             "Performance": f"Best balance. Current: AUC={new_auc:.4f}"},
        ]

        with mlflow.start_run(run_name="Retraining_Strategy"):
            mlflow.log_param("trigger_type", trigger)
            mlflow.log_param("trigger_reason", reason)
            mlflow.log_metric("previous_auc", current_auc)
            mlflow.log_metric("new_auc", new_auc)
            mlflow.log_metric("new_recall", new_recall)
            mlflow.log_metric("auc_improvement", improvement)
            mlflow.log_metric("retrain_time_seconds", train_time)

            # Save strategy comparison
            comp_df = pd.DataFrame(strategy_results)
            comp_path = os.path.join(INTERIM_DIR, "retraining_strategy_comparison.csv")
            comp_df.to_csv(comp_path, index=False)
            mlflow.log_artifact(comp_path)
            print("\n  Strategy Comparison:")
            print(comp_df.to_string(index=False))
    else:
        print("  → No retraining needed. Model is stable.")
        with mlflow.start_run(run_name="Retraining_Check_Skipped"):
            mlflow.log_param("decision", "skip")
            mlflow.log_param("reason", reason)

    print("\n  ✓ Retraining strategy check complete.")


if __name__ == "__main__":
    run_retraining_strategy()
