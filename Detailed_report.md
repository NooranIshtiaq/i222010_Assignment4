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
We compared three strategies to handle the extreme class imbalance (only ~3.5% fraud cases).

![Model Comparison](screenshots/model_comparison.JPG)

### 3.2 Business Impact (Task 4)
Cost-sensitive training significantly reduced "False Negatives," saving more money despite a slight increase in "False Alarms."

![Cost-Sensitive Performance](screenshots/xgboost_cost_sensitive.JPG)

## 4. Task 3: Model Complexity
We evaluated three different architectures. Performance metrics for LightGBM and XGBoost variants are shown below:

![LightGBM Performance](screenshots/lightGBM_baseline.JPG)

## 5. Task 5: CI/CD Pipeline
Implemented using **GitHub Actions**.

![GitHub Actions Workflow](screenshots/github_ci_cd.JPG)

## 6. Task 6: Observability & Monitoring
A comprehensive monitoring layer was built using **Prometheus** and **Grafana**.

![Model Performance Dashboard](screenshots/model_performance_dash1.JPG)
![Data Drift Dashboard](screenshots/data_drift_dashboard.JPG)

## 7. Task 7 & 8: Drift Simulation & Retraining
We simulated **Time-based Drift** and introduced new fraud patterns.

![Prometheus Alerts](screenshots/promethus_alerts.JPG)

## 8. Task 9: Model Explainability (SHAP)
Using **SHAP**, we identified the top features driving fraud predictions.

![SHAP Summary Plot](screenshots/shap_summary.JPG)
![SHAP Waterfall Plot](screenshots/shap_waterfall.JPG)

---
## 9. Final Conclusion
The system successfully meets all assignment criteria, providing a production-ready solution that balances model accuracy with operational efficiency and monitoring.
