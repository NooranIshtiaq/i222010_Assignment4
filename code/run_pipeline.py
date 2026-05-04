"""
Main Pipeline Orchestrator — Replaces Airflow DAG
Runs all pipeline steps sequentially with conditional logic and retry mechanisms.
"""
import subprocess
import sys
import json
import os
import time

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "src")
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

# ---------- Configuration ----------
AUC_THRESHOLD = 0.80
RECALL_THRESHOLD = 0.85
MAX_RETRIES = 2
RETRY_DELAY_SEC = 5


def run_step(script_name, step_label, retries=MAX_RETRIES):
    """Run a pipeline step with retry logic."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    for attempt in range(1, retries + 1):
        print(f"\n{'='*60}")
        print(f"  STEP: {step_label}  (attempt {attempt}/{retries})")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            print(f"  ✓ {step_label} completed successfully.")
            return True
        print(f"  ✗ {step_label} failed (exit code {result.returncode}).")
        if attempt < retries:
            print(f"    Retrying in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)
    print(f"  ✗✗ {step_label} FAILED after {retries} attempts. Aborting pipeline.")
    sys.exit(1)


def check_deployment_condition():
    """Conditional deployment — deploy only if AUC > threshold."""
    auc_file = os.path.join(INTERIM_DIR, "best_auc.json")
    try:
        with open(auc_file, "r") as f:
            best_auc = json.load(f).get("best_auc", 0)
    except FileNotFoundError:
        best_auc = 0
    return best_auc


def main():
    start = time.time()
    print("\n" + "=" * 60)
    print("  FRAUD DETECTION ML PIPELINE")
    print("  Orchestrator: Python  |  Tracking: MLflow")
    print("=" * 60)

    # ---- Core Pipeline (Tasks 1-3) ----
    run_step("1_ingest_data.py", "1 · Data Ingestion")
    run_step("2_validate_data.py", "2 · Data Validation")
    run_step("3_preprocess_data.py", "3 · Data Preprocessing")
    run_step("4_feature_engineering.py", "4 · Feature Engineering")
    run_step("5_train_models.py", "5 · Model Training (XGB, LGBM, Hybrid, SMOTE, Cost-Sensitive)")
    run_step("6_evaluate_model.py", "6 · Model Evaluation (Metrics + SHAP + Feature Importance)")
    run_step("7_register_model.py", "7 · Model Registration")

    # ---- Drift & Retraining (Tasks 7-8) ----
    run_step("8_drift_simulation.py", "8 · Drift Simulation")
    run_step("9_retraining_strategy.py", "9 · Intelligent Retraining Check")

    # ---- Business Impact (Task 4) ----
    run_step("10_business_impact.py", "10 · Business Impact Analysis")

    # ---- Conditional Deployment (Task 1 requirement) ----
    best_auc = check_deployment_condition()
    print(f"\n{'='*60}")
    print(f"  CONDITIONAL DEPLOYMENT CHECK")
    print(f"  Best AUC: {best_auc:.4f}  |  Threshold: {AUC_THRESHOLD}")
    print(f"{'='*60}")
    if best_auc > AUC_THRESHOLD:
        print("  ✓ AUC exceeds threshold → Model DEPLOYED.")
    else:
        print("  ✗ AUC below threshold → Deployment SKIPPED.")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
