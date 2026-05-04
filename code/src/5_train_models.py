"""
Step 5: Model Training (Tasks 2, 3, 4)
- XGBoost (Standard, Cost-Sensitive, SMOTE)
- LightGBM baseline
- Hybrid: RF + SelectFromModel feature selection
- Imbalance comparison: SMOTE vs Undersampling vs Class Weighting
- All logged to MLflow
"""
import mlflow
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# DATA_DIR = os.path.join(BASE_DIR, "Data")
# INTERIM_DIR = os.path.join(DATA_DIR, "interim")
DATA_DIR = "/app/Data"
INTERIM_DIR = "/app/Data/interim"


MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")


def train_models():
    print("--- Step 5: Model Training ---")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Fraud_Detection_Pipeline")

    # Load data
    df = pd.read_csv(os.path.join(INTERIM_DIR, "04_engineered.csv"))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df.drop(["isFraud", "TransactionID"], axis=1, errors="ignore")
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test sets for evaluation script
    test_dir = os.path.join(INTERIM_DIR, "test_data")
    os.makedirs(test_dir, exist_ok=True)
    X_test.to_csv(os.path.join(test_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(test_dir, "y_test.csv"), index=False)

    # ---- Define all experiments ----
    experiments = [
        # Task 2 & 4: Imbalance strategies + Cost-Sensitive
        {"name": "1_XGBoost_Standard",
         "model": XGBClassifier(eval_metric="logloss", random_state=42),
         "sampler": None, "sampler_name": "None"},

        {"name": "2_XGBoost_Cost_Sensitive",
         "model": XGBClassifier(scale_pos_weight=10, eval_metric="logloss", random_state=42),
         "sampler": None, "sampler_name": "ClassWeighting(10x)"},

        {"name": "3_XGBoost_SMOTE",
         "model": XGBClassifier(eval_metric="logloss", random_state=42),
         "sampler": SMOTE(random_state=42), "sampler_name": "SMOTE"},

        {"name": "4_XGBoost_Undersampling",
         "model": XGBClassifier(eval_metric="logloss", random_state=42),
         "sampler": RandomUnderSampler(random_state=42), "sampler_name": "RandomUnderSampler"},

        # Task 3: LightGBM
        {"name": "5_LightGBM_Baseline",
         "model": LGBMClassifier(verbose=-1, random_state=42),
         "sampler": None, "sampler_name": "None"},

        # Task 3: Hybrid (RF + Feature Selection)
        {"name": "6_Hybrid_RF_SelectFromModel",
         "model": Pipeline([
             ("feature_selection", SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42))),
             ("classification", RandomForestClassifier(n_estimators=100, random_state=42))
         ]),
         "sampler": None, "sampler_name": "None"},
    ]

    comparison_rows = []

    for exp in experiments:
        print(f"\n  Training: {exp['name']} (sampler: {exp['sampler_name']})")
        with mlflow.start_run(run_name=exp["name"]):
            curr_X, curr_y = X_train.copy(), y_train.copy()

            # Apply sampling strategy
            if exp["sampler"] is not None:
                curr_X, curr_y = exp["sampler"].fit_resample(X_train, y_train)
                print(f"    Resampled: {len(curr_y)} samples (fraud ratio: {curr_y.mean():.2%})")

            model = exp["model"]
            model.fit(curr_X, curr_y)

            # Evaluate
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, proba)

            # Log to MLflow
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_param("sampler", exp["sampler_name"])
            mlflow.log_param("cost_sensitive", "Cost_Sensitive" in exp["name"])
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc)

            print(f"    → AUC: {auc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            # Save model locally
            model_path = os.path.join(INTERIM_DIR, f"{exp['name']}.pkl")
            joblib.dump(model, model_path)

            # Collect for comparison
            comparison_rows.append({
                "Model": exp["name"],
                "Sampler": exp["sampler_name"],
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
                "AUC_ROC": round(auc, 4),
            })

    # ---- Task 2: Imbalance Strategy Comparison Artifact ----
    comparison_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(INTERIM_DIR, "imbalance_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"\n  ✓ Imbalance comparison saved: {comp_path}")
    print(comparison_df.to_string(index=False))

    # Log comparison to MLflow
    with mlflow.start_run(run_name="Imbalance_Strategy_Comparison"):
        mlflow.log_artifact(comp_path)

    print("\n  ✓ All models trained and logged to MLflow.")


if __name__ == "__main__":
    train_models()
