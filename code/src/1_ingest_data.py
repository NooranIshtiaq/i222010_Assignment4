"""
Step 1: Data Ingestion
- Loads train_transaction.csv (subset for speed)
- Merges with train_identity.csv if available
- Saves raw data to interim directory
"""

import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")


def ingest():
    print("--- Step 1: Data Ingestion ---")
    os.makedirs(INTERIM_DIR, exist_ok=True)

    txn_path = os.path.join(DATA_DIR, "train_transaction.csv")

    # ✅ LOAD ONLY 10,000 ROWS
    df_txn = pd.read_csv(txn_path, nrows=10000)

    print(f"  Loaded {len(df_txn)} transaction rows, {df_txn.shape[1]} columns")

    id_path = os.path.join(DATA_DIR, "train_identity.csv")

    if os.path.exists(id_path):
        df_id = pd.read_csv(id_path)

        df = df_txn.merge(df_id, on="TransactionID", how="left")
        print(f"  Merged with identity data → {df.shape[1]} columns")
    else:
        df = df_txn
        print("  Identity data not found, using transactions only.")

    out_path = os.path.join(INTERIM_DIR, "01_raw.csv")
    df.to_csv(out_path, index=False)

    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    ingest()
