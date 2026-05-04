"""
Step 8: Drift Simulation (Task 7)
- Time-based drift: train on earlier data, test on later distribution
- Inject new fraud patterns
- Compare feature importance shifts between periods
- Log drift metrics to MLflow
"""

import mlflow
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score

DATA_DIR = "/app/Data"
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "plots")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
DRIFT_SPLIT_RATIO = 0.6


def simulate_drift():
    print("--- Step 8: Drift Simulation (Task 7) ---")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Fraud_Detection_Pipeline")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(INTERIM_DIR, exist_ok=True)

    txn_path = os.path.join(DATA_DIR, "train_transaction.csv")

    print("  Loading chronological data...")
    df = pd.read_csv(txn_path, nrows=10000)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols].fillna(0)

    if "TransactionDT" not in df.columns:
        print("  ✗ TransactionDT not found. Skipping drift simulation.")
        return

    df = df.sort_values("TransactionDT")

    split_idx = int(len(df) * DRIFT_SPLIT_RATIO)
    early_df = df.iloc[:split_idx].copy()
    late_df = df.iloc[split_idx:].copy()

    print(f"  Early period: {len(early_df)} rows | Late period: {len(late_df)} rows")

    feature_cols = [c for c in numeric_cols if c not in ["isFraud", "TransactionID", "TransactionDT"]]

    # ---------------- TRAIN ----------------
    X_early = early_df[feature_cols]
    y_early = early_df["isFraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_early, y_early, test_size=0.2, random_state=42
    )

    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    preds_baseline = model.predict(X_val)
    proba_baseline = model.predict_proba(X_val)[:, 1]

    recall_baseline = recall_score(y_val, preds_baseline, zero_division=0)
    auc_baseline = roc_auc_score(y_val, proba_baseline)

    print(f"\n  Baseline (early data): Recall={recall_baseline:.4f}, AUC={auc_baseline:.4f}")

    # ---------------- DRIFT INJECTION ----------------
    late_drifted = late_df.copy()

    if "card1" in late_drifted.columns:
        mask = late_drifted["card1"] > late_drifted["card1"].quantile(0.9)
        late_drifted.loc[mask, "isFraud"] = 1
        print(f"  Injected {mask.sum()} new fraud patterns (high card1)")

    if "TransactionAmt" in late_drifted.columns:
        late_drifted["TransactionAmt"] = late_drifted["TransactionAmt"] * 1.5 + 50

    X_late = late_drifted[feature_cols]
    y_late = late_drifted["isFraud"]

    preds_drift = model.predict(X_late)
    proba_drift = model.predict_proba(X_late)[:, 1]

    recall_drift = recall_score(y_late, preds_drift, zero_division=0)
    auc_drift = roc_auc_score(y_late, proba_drift)

    print(f"  Drifted (late data): Recall={recall_drift:.4f}, AUC={auc_drift:.4f}")

    recall_delta = float(recall_drift - recall_baseline)
    auc_delta = float(auc_drift - auc_baseline)

    print(f"  Δ Recall: {recall_delta:+.4f} | Δ AUC: {auc_delta:+.4f}")

    # ---------------- FEATURE IMPORTANCE ----------------
    importance_early = pd.Series(model.feature_importances_, index=feature_cols)

    model_late = XGBClassifier(eval_metric="logloss", random_state=42)
    X_lt, _, y_lt, _ = train_test_split(X_late, y_late, test_size=0.2, random_state=42)
    model_late.fit(X_lt, y_lt)

    importance_late = pd.Series(model_late.feature_importances_, index=feature_cols)

    top_features = importance_early.sort_values(ascending=False).head(10).index

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    importance_early[top_features].plot(kind="barh", ax=axes[0], title="Early Period")
    importance_late[top_features].plot(kind="barh", ax=axes[1], title="Late Period (Drifted)")
    fig.suptitle("Feature Importance Shift")

    shift_path = os.path.join(ARTIFACTS_DIR, "drift_feature_importance_shift.png")
    plt.savefig(shift_path, bbox_inches="tight", dpi=100)
    plt.close()

    # ---------------- PERFORMANCE PLOT ----------------
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Recall", "AUC"]
    baseline_vals = [recall_baseline, auc_baseline]
    drifted_vals = [recall_drift, auc_drift]

    x = np.arange(len(metrics))

    ax.bar(x - 0.2, baseline_vals, 0.35, label="Baseline")
    ax.bar(x + 0.2, drifted_vals, 0.35, label="Drifted")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()

    perf_path = os.path.join(ARTIFACTS_DIR, "drift_performance.png")
    plt.savefig(perf_path, bbox_inches="tight", dpi=100)
    plt.close()

    # ---------------- MLflow ----------------
    with mlflow.start_run(run_name="Drift_Simulation"):
        mlflow.log_metric("baseline_recall", float(recall_baseline))
        mlflow.log_metric("baseline_auc", float(auc_baseline))
        mlflow.log_metric("drifted_recall", float(recall_drift))
        mlflow.log_metric("drifted_auc", float(auc_drift))
        mlflow.log_metric("recall_delta", recall_delta)
        mlflow.log_metric("auc_delta", auc_delta)

        mlflow.log_artifact(shift_path)
        mlflow.log_artifact(perf_path)

    # ---------------- JSON SAFE REPORT ----------------
    drift_report = {
        "baseline_recall": float(recall_baseline),
        "baseline_auc": float(auc_baseline),
        "drifted_recall": float(recall_drift),
        "drifted_auc": float(auc_drift),
        "recall_delta": recall_delta,
        "auc_delta": auc_delta,
        "drift_detected": bool(abs(recall_delta) > 0.05 or abs(auc_delta) > 0.05),
    }

    report_path = os.path.join(INTERIM_DIR, "drift_report.json")

    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    status = "DETECTED ⚠️" if drift_report["drift_detected"] else "NOT DETECTED ✓"
    print(f"\n  Drift Status: {status}")
    print("  ✓ Drift simulation complete.")


if __name__ == "__main__":
    simulate_drift()