"""
Test script — sends a sample prediction request to the Inference API.
Usage: python send_test.py
"""
import pandas as pd
import requests
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "interim", "04_engineered.csv")

try:
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    to_drop = ["isFraud", "TransactionID"]
    sample_row = df.drop(columns=[c for c in to_drop if c in df.columns]).iloc[0].to_dict()

    # Convert numpy types to JSON-safe Python types
    clean_row = {k: (v.item() if hasattr(v, "item") else v) for k, v in sample_row.items()}

    url = "http://localhost:8001/predict"
    response = requests.post(url, json=clean_row)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

except Exception as e:
    print(f"Error: {e}")
