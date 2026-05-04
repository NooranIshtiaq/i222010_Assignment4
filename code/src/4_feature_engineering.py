"""
Step 4: Feature Engineering (Task 2 — Data Challenges)
- Target Encoding for high-cardinality categorical features
- Saves encoder for inference API reuse
"""
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import joblib
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "encoders")


def engineer():
    print("--- Step 4: Feature Engineering ---")
    in_path = os.path.join(INTERIM_DIR, "03_preprocessed.csv")
    df = pd.read_csv(in_path)

    # 1. Identify all remaining object (categorical) columns
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 2. Add known high-cardinality numeric columns
    extra_high_card = ["card1", "card2", "addr1", "addr2"]
    cols_to_encode = list(set(
        object_cols + [c for c in extra_high_card if c in df.columns]
    ))

    # 3. Target Encode all of them
    if cols_to_encode:
        print(f"  Target Encoding {len(cols_to_encode)} columns: {cols_to_encode[:5]}...")
        te = TargetEncoder(cols=cols_to_encode)
        df[cols_to_encode] = te.fit_transform(df[cols_to_encode], df["isFraud"])

        # Save encoder for inference API
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        enc_path = os.path.join(ARTIFACTS_DIR, "target_encoder.pkl")
        joblib.dump(te, enc_path)
        print(f"  Encoder saved to {enc_path}")

    # 4. Replace any infinities created during encoding
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    out_path = os.path.join(INTERIM_DIR, "04_engineered.csv")
    df.to_csv(out_path, index=False)
    print(f"  ✓ Feature Engineering complete → {df.shape[1]} features")


if __name__ == "__main__":
    engineer()
