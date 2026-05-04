"""
Step 7: Model Registration
- Finds best model from evaluation results
- Registers it as the champion model in MLflow Model Registry
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import json
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def register():
    print("--- Step 7: Model Registration ---")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Fraud_Detection_Pipeline")

    # Find the best model
    auc_path = os.path.join(INTERIM_DIR, "best_auc.json")
    try:
        with open(auc_path, "r") as f:
            info = json.load(f)
            best_model_name = info.get("best_model", "2_XGBoost_Cost_Sensitive")
            best_auc = info.get("best_auc", 0)
    except FileNotFoundError:
        best_model_name = "2_XGBoost_Cost_Sensitive"
        best_auc = 0

    model_path = os.path.join(INTERIM_DIR, f"{best_model_name}.pkl")
    if not os.path.exists(model_path):
        print(f"  ✗ Model file not found: {model_path}")
        return

    model = joblib.load(model_path)
    print(f"  Registering champion: {best_model_name} (AUC={best_auc:.4f})")

    with mlflow.start_run(run_name="Champion_Registration"):
        mlflow.log_param("champion_model", best_model_name)
        mlflow.log_metric("champion_auc", best_auc)

        # Register based on model type
        model_type = type(model).__name__
        if "XGB" in model_type:
            mlflow.xgboost.log_model(model, "model", registered_model_name="FraudModelV1")
        else:
            mlflow.sklearn.log_model(model, "model", registered_model_name="FraudModelV1")

        print("  ✓ Champion model registered as FraudModelV1 in MLflow Registry.")


if __name__ == "__main__":
    register()
