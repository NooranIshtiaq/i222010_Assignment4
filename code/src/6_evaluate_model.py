"""
Step 6: Model Evaluation (Tasks 3, 4, 9)
- Precision, Recall, F1, AUC-ROC for all models
- Confusion matrix with TP/FP/FN/TN for fraud class
- SHAP explainability (summary + individual force plots)
- Feature importance bar charts
- All artifacts logged to MLflow
"""
import mlflow
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "plots")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

MODEL_NAMES = [
    "1_XGBoost_Standard",
    "2_XGBoost_Cost_Sensitive",
    "3_XGBoost_SMOTE",
    "4_XGBoost_Undersampling",
    "5_LightGBM_Baseline",
    "6_Hybrid_RF_SelectFromModel",
]


def evaluate():
    print("--- Step 6: Model Evaluation ---")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_test = pd.read_csv(os.path.join(INTERIM_DIR, "test_data", "X_test.csv"))
    y_test = pd.read_csv(os.path.join(INTERIM_DIR, "test_data", "y_test.csv"))["isFraud"]

    best_auc = 0
    best_model_name = ""
    all_results = []
    eval_details = {}  # For business impact script

    for model_name in MODEL_NAMES:
        model_path = os.path.join(INTERIM_DIR, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            print(f"  Skipping {model_name} — file not found.")
            continue

        model = joblib.load(model_path)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

        # Metrics
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, proba)

        # Confusion matrix — fraud class details
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"\n  {model_name}:")
        print(f"    AUC: {auc:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")
        print(f"    Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn} | FPR: {fpr:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name

        # Store for business impact
        eval_details[model_name] = {
            "precision": precision, "recall": recall, "f1": f1, "auc": auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "fpr": fpr,
        }

        all_results.append({
            "Model": model_name, "Precision": round(precision, 4),
            "Recall": round(recall, 4), "F1": round(f1, 4),
            "AUC_ROC": round(auc, 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })

        with mlflow.start_run(run_name=f"Eval_{model_name}"):
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc)
            mlflow.log_metric("true_positives", tp)
            mlflow.log_metric("false_positives", fp)
            mlflow.log_metric("false_negatives", fn)
            mlflow.log_metric("true_negatives", tn)
            mlflow.log_metric("false_positive_rate", fpr)

            # ---- Confusion Matrix CSV ----
            cm_path = os.path.join(ARTIFACTS_DIR, f"cm_{model_name}.csv")
            pd.DataFrame(
                confusion_matrix(y_test, preds),
                columns=["Pred_Legit", "Pred_Fraud"],
                index=["True_Legit", "True_Fraud"],
            ).to_csv(cm_path)
            mlflow.log_artifact(cm_path)

            # ---- ROC Curve Plot ----
            fpr_arr, tpr_arr, _ = roc_curve(y_test, proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr_arr, tpr_arr, label=f"AUC={auc:.4f}")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve — {model_name}")
            plt.legend()
            roc_path = os.path.join(ARTIFACTS_DIR, f"roc_{model_name}.png")
            plt.savefig(roc_path, bbox_inches="tight", dpi=100)
            plt.close()
            mlflow.log_artifact(roc_path)

            # ---- Task 9: SHAP Explainability ----
            if "Hybrid" not in model_name:
                try:
                    print("    Generating SHAP analysis...")
                    # Use a subset for speed
                    X_shap = X_test.iloc[:200]

                    if "XGBoost" in model_name or "LightGBM" in model_name:
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_shap)

                    # SHAP Summary Plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
                    shap_path = os.path.join(ARTIFACTS_DIR, f"shap_summary_{model_name}.png")
                    plt.savefig(shap_path, bbox_inches="tight", dpi=100)
                    plt.close()
                    mlflow.log_artifact(shap_path)

                    # Feature Importance Bar Chart
                    if hasattr(model, "feature_importances_"):
                        importance = pd.Series(
                            model.feature_importances_, index=X_test.columns
                        ).sort_values(ascending=False).head(15)
                        plt.figure(figsize=(10, 6))
                        importance.plot(kind="barh", color="steelblue")
                        plt.title(f"Feature Importance — {model_name}")
                        plt.xlabel("Importance")
                        plt.gca().invert_yaxis()
                        fi_path = os.path.join(ARTIFACTS_DIR, f"feat_importance_{model_name}.png")
                        plt.savefig(fi_path, bbox_inches="tight", dpi=100)
                        plt.close()
                        mlflow.log_artifact(fi_path)

                    # SHAP Force Plot — Top 3 fraud predictions (individual explanations)
                    fraud_indices = y_test[y_test == 1].index[:3]
                    for i, idx in enumerate(fraud_indices):
                        if idx < len(X_shap):
                            plt.figure(figsize=(14, 3))
                            shap.force_plot(
                                explainer.expected_value if isinstance(explainer.expected_value, float)
                                else explainer.expected_value[1],
                                shap_values[idx] if len(np.array(shap_values).shape) == 2
                                else np.array(shap_values)[1][idx],
                                X_shap.iloc[idx],
                                matplotlib=True, show=False,
                            )
                            fp_path = os.path.join(ARTIFACTS_DIR, f"shap_force_{model_name}_sample{i}.png")
                            plt.savefig(fp_path, bbox_inches="tight", dpi=100)
                            plt.close()
                            mlflow.log_artifact(fp_path)
                except Exception as e:
                    print(f"    SHAP failed for {model_name}: {e}")

    # ---- Consolidated comparison table ----
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(INTERIM_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print("\n  Model Comparison:")
    print(results_df.to_string(index=False))

    with mlflow.start_run(run_name="Model_Comparison"):
        mlflow.log_artifact(results_path)

    # ---- Save best AUC for conditional deployment ----
    auc_path = os.path.join(INTERIM_DIR, "best_auc.json")
    with open(auc_path, "w") as f:
        json.dump({"best_auc": best_auc, "best_model": best_model_name}, f)
    print(f"\n  ✓ Best model: {best_model_name} (AUC={best_auc:.4f})")

    # ---- Save eval details for business impact ----
    details_path = os.path.join(INTERIM_DIR, "eval_details.json")
    with open(details_path, "w") as f:
        json.dump(eval_details, f, indent=2)


if __name__ == "__main__":
    evaluate()
