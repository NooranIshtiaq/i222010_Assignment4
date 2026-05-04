import requests
import time
import random
import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "interim", "04_engineered.csv")
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
to_drop = ["isFraud", "TransactionID"]
clean_df = df.drop(columns=[c for c in to_drop if c in df.columns])

url = "http://localhost:8001/predict"

print("Starting live traffic simulation... Press Ctrl+C to stop.")
try:
    for i in range(10000):
        # Pick a random row
        row = clean_df.sample(1).iloc[0].to_dict()
        
        # 10% chance to send an anomaly (tons of nulls) to trigger anomaly alerts
        if random.random() < 0.10:
            for k in list(row.keys())[:int(len(row)*0.4)]:
                row[k] = None
                
        # 5% chance to simulate a massive latency spike
        if random.random() < 0.05:
            time.sleep(random.uniform(1.0, 3.0))
            
        clean_row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        
        try:
            requests.post(url, json=clean_row)
            print(f"Sent request {i+1}...", end="\r")
        except Exception as e:
            pass
        
        # Normal traffic delay
        time.sleep(random.uniform(0.1, 0.5))
        
except KeyboardInterrupt:
    print("\nStopped.")
