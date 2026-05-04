"""
Step 2: Data Validation
- Schema checks (required columns exist)
- Missing value thresholds
- Data integrity checks
"""
import pandas as pd
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

REQUIRED_COLUMNS = ["isFraud", "TransactionAmt", "TransactionDT"]


def validate():
    print("--- Step 2: Data Validation ---")
    in_path = os.path.join(INTERIM_DIR, "01_raw.csv")
    df = pd.read_csv(in_path)

    errors = []

    # 1. Schema check — required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Required column '{col}' is missing.")

    # 2. Target column integrity
    if "isFraud" in df.columns:
        if df["isFraud"].isnull().any():
            errors.append("Target column 'isFraud' contains null values.")
        if not set(df["isFraud"].dropna().unique()).issubset({0, 1}):
            errors.append("Target column 'isFraud' has values other than 0/1.")

    # 3. Transaction amount sanity
    if "TransactionAmt" in df.columns:
        if (df["TransactionAmt"] < 0).any():
            errors.append("Negative transaction amounts found.")

    # 4. Overall missing value threshold
    overall_missing = df.isnull().mean().mean()
    print(f"  Overall missing rate: {overall_missing:.2%}")
    if overall_missing > 0.80:
        errors.append(f"Too many missing values ({overall_missing:.2%} > 80%).")

    # 5. Row count sanity
    if len(df) < 100:
        errors.append(f"Too few rows ({len(df)}). Minimum expected: 100.")

    if errors:
        print("  ✗ VALIDATION FAILED:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print(f"  ✓ All checks passed ({len(df)} rows, {df.shape[1]} columns)")

    out_path = os.path.join(INTERIM_DIR, "02_validated.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    validate()
