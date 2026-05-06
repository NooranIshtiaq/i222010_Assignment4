import requests
import time
import pandas as pd
import numpy as np
import os
import random

# Path to the processed data
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "interim", "04_engineered.csv")

def send_bulk(n=50):
    print(f"--- Sending {n} test cases to Inference API (50/50 Fraud Mix) ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Separate fraud and non-fraud to ensure a mix
    fraud_df = df[df["isFraud"] == 1]
    non_fraud_df = df[df["isFraud"] == 0]
    
    to_drop = ["isFraud", "TransactionID"]
    
    url = "http://localhost:8001/predict"
    
    for i in range(n):
        # 50% chance to pick a fraud case
        if i % 2 == 0 and not fraud_df.empty:
            sample_row = fraud_df.sample(1).iloc[0].to_dict()
            label = "FRAUD"
        else:
            sample_row = non_fraud_df.sample(1).iloc[0].to_dict()
            label = "NORMAL"
            
        clean_row = {k: (v.item() if hasattr(v, "item") else v) for k, v in sample_row.items() if k not in to_drop}
        
        try:
            response = requests.post(url, json=clean_row)
            res_json = response.json()
            pred = res_json.get('is_fraud')
            conf = res_json.get('confidence')
            print(f"[{i+1}/{n}] Actual: {label} | Prediction: {pred} | Confidence: {conf}")
        except Exception as e:
            print(f"[{i+1}/{n}] Failed: {e}")
        
        time.sleep(0.1)

if __name__ == "__main__":
    send_bulk(50)
