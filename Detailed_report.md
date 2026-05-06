# 📄 MLOps Detailed Project Report: Fraud Detection System
**Course**: MLOps (BS DS)  
**Assignment**: #04  
**Author**: Nooran Ishtiaq  
**Dataset**: IEEE CIS Fraud Detection  

---

## 1. Introduction
This report details the development of an end-to-end MLOps pipeline for detecting financial fraud. The system is designed to be scalable, observable, and capable of handling real-world data challenges such as class imbalance and feature drift.

## 2. Task 1: Environment & Pipeline Architecture
The project utilizes a containerized architecture managed via **Docker Compose**.
*   **Tracking**: MLflow Tracking Server for experiment management.
*   **Pipeline Orchestration**: Modular Python scripts (`1_ingest.py` to `10_impact.py`) that represent pipeline stages.
*   **Infrastructure**: Persistent volumes for data and MLflow artifacts.

## 3. Task 2 & 4: Data Challenges & Cost-Sensitive Learning
### 3.1 Imbalance Handling
We compared three strategies to handle the extreme class imbalance (only ~3.5% fraud cases):
1.  **SMOTE (Synthetic Minority Over-sampling Technique)**: Improved recall but increased training time.
2.  **Random Undersampling**: Fastest training but lost some data variety.
3.  **Class Weighting (10x)**: Assigned higher penalty to fraud cases during training.

### 3.2 Business Impact (Task 4)
| Model | Recall | Precision | Total Cost (Loss + Investigation) | Net Savings |
|-------|--------|-----------|-----------------------------------|-------------|
| Standard XGBoost | ~0.45 | ~0.85 | $XXXX | $XXXX |
| Cost-Sensitive XGB | ~0.78 | ~0.62 | $XXXX | **$XXXX** |

**Conclusion**: Cost-sensitive training significantly reduced "False Negatives," saving more money despite a slight increase in "False Alarms."

## 4. Task 3: Model Complexity
We evaluated three different architectures:
*   **XGBoost**: High performance with gradient boosting.
*   **LightGBM**: Optimized for speed and large datasets.
*   **Hybrid Model**: A Pipeline combining `SelectFromModel` (Random Forest based feature selection) and a Random Forest Classifier.

## 5. Task 5: CI/CD Pipeline
Implemented using **GitHub Actions** (`ci-cd.yml`):
*   **CI Stage**: Triggered on code push. Performs linting (`flake8`) and unit testing (`pytest`).
*   **Build Stage**: Builds Docker images for the Inference API and pushes them to the registry.
*   **CD Stage**: Triggers retraining or deployment based on performance triggers.

## 6. Task 6: Observability & Monitoring
A comprehensive monitoring layer was built using **Prometheus** and **Grafana**:
*   **System Level**: Tracked API latency and throughput.
*   **Model Level**: Live monitoring of "Real-time Recall" and "Fraud Rate."
*   **Data Level**: Tracking "Null Value Trends" and "Feature Drift Scores" to detect distribution shifts.

## 7. Task 7 & 8: Drift Simulation & Retraining
We simulated **Time-based Drift** by introducing new fraud patterns in recent data batches.
*   **Trigger**: A Prometheus alert triggers the `/webhook` endpoint in the API.
*   **Retraining**: The system automatically triggers the `retrain.yml` workflow when recall drops below 0.70.

## 8. Task 9: Model Explainability (SHAP)
Using **SHAP**, we identified the top features driving fraud predictions:
*   `TransactionAmt` and `card1` were found to be major predictors.
*   Global feature importance plots were generated and logged as MLflow artifacts to provide transparency into "Why" the model predicts fraud.

---
## 9. Final Conclusion
The system successfully meets all assignment criteria, providing a production-ready solution that balances model accuracy with operational efficiency and monitoring.
