"""
Step 10: Business Impact Analysis (Task 4)
- Compares standard vs cost-sensitive training
- Assigns dollar values to FN (missed fraud) and FP (false alarms)
"""
import mlflow
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# DATA_DIR = os.path.join(BASE_DIR, "Data")
# INTERIM_DIR = os.path.join(DATA_DIR, "interim")
DATA_DIR = "/app/Data"
INTERIM_DIR = "/app/Data/interim"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "plots")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

AVG_FRAUD_AMOUNT = 150.0
INVESTIGATION_COST = 50.0


def analyze_business_impact():
    print("--- Step 10: Business Impact Analysis (Task 4) ---")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    details_path = os.path.join(INTERIM_DIR, "eval_details.json")
    try:
        with open(details_path, "r") as f:
            eval_details = json.load(f)
    except FileNotFoundError:
        print("  eval_details.json not found. Run evaluation first.")
        return

    models_to_compare = {
        "1_XGBoost_Standard": "Standard Training",
        "2_XGBoost_Cost_Sensitive": "Cost-Sensitive (10x FN penalty)",
    }

    results = []
    for model_key, label in models_to_compare.items():
        if model_key not in eval_details:
            continue
        m = eval_details[model_key]
        fn, fp, tp = m["fn"], m["fp"], m["tp"]
        fraud_loss = fn * AVG_FRAUD_AMOUNT
        false_alarm_cost = fp * INVESTIGATION_COST
        total_cost = fraud_loss + false_alarm_cost
        fraud_caught_value = tp * AVG_FRAUD_AMOUNT

        results.append({
            "Model": label, "Recall": m["recall"], "Precision": m["precision"],
            "FN_Missed_Fraud": fn, "FP_False_Alarms": fp,
            "Fraud_Loss": round(fraud_loss, 2),
            "Investigation_Cost": round(false_alarm_cost, 2),
            "Total_Cost": round(total_cost, 2),
            "Net_Savings": round(fraud_caught_value - total_cost, 2),
        })

    if not results:
        print("  No models found for comparison.")
        return

    df = pd.DataFrame(results)
    print("\n  Business Impact Comparison:")
    print(df.to_string(index=False))

    report_path = os.path.join(INTERIM_DIR, "business_impact_report.csv")
    df.to_csv(report_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(df))
    w = 0.3
    axes[0].bar([i - w / 2 for i in x], df["Fraud_Loss"], w, label="Fraud Loss", color="crimson")
    axes[0].bar([i + w / 2 for i in x], df["Investigation_Cost"], w, label="Investigation Cost", color="orange")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(df["Model"], fontsize=8)
    axes[0].set_ylabel("Cost ($)")
    axes[0].set_title("Cost Breakdown")
    axes[0].legend()
    colors = ["steelblue" if v >= 0 else "crimson" for v in df["Net_Savings"]]
    axes[1].bar(df["Model"], df["Net_Savings"], color=colors)
    axes[1].set_ylabel("Net Savings ($)")
    axes[1].set_title("Net Savings")
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    chart_path = os.path.join(ARTIFACTS_DIR, "business_impact_comparison.png")
    plt.savefig(chart_path, bbox_inches="tight", dpi=100)
    plt.close()

    with mlflow.start_run(run_name="Business_Impact_Analysis"):
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(chart_path)

    print("\n  Business impact report saved.")


if __name__ == "__main__":
    analyze_business_impact()
