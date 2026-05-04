"""
Step 3: Data Preprocessing (Task 2 — Data Challenges)
- Advanced missing value handling: KNN Imputer for numeric columns
- Median fallback for remaining NaNs
- Drop columns with extreme missingness (>90%)
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os

#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#DATA_DIR = os.path.join(BASE_DIR, "Data")
#INTERIM_DIR = os.path.join(DATA_DIR, "interim")
DATA_DIR = "/app/Data"
INTERIM_DIR = "/app/Data/interim"

MISSINGNESS_THRESHOLD = 0.90  # Drop columns with >90% missing
KNN_NEIGHBORS = 5
MAX_KNN_COLS = 20  # Limit KNN imputation to top N cols for speed


def preprocess():
    print("--- Step 3: Data Preprocessing ---")
    in_path = os.path.join(INTERIM_DIR, "02_validated.csv")
    df = pd.read_csv(in_path)

    # 1. Drop columns with extreme missingness
    missing_rates = df.isnull().mean()
    drop_cols = missing_rates[missing_rates > MISSINGNESS_THRESHOLD].index.tolist()
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} columns with >{MISSINGNESS_THRESHOLD*100:.0f}% missing")
        df = df.drop(columns=drop_cols)

    # 2. KNN Imputation on numeric columns (excluding target & ID)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    protected = ["isFraud", "TransactionID"]
    cols_for_knn = [c for c in numeric_cols if c not in protected]

    # Only KNN-impute columns that have missing values, limit count for speed
    cols_with_missing = [c for c in cols_for_knn if df[c].isnull().any()]
    cols_to_impute = cols_with_missing[:MAX_KNN_COLS]

    if cols_to_impute:
        print(f"  KNN Imputing {len(cols_to_impute)} columns (k={KNN_NEIGHBORS})...")
        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    # 3. Median fallback for any remaining numeric NaNs
    remaining_numeric = df.select_dtypes(include=[np.number]).columns
    remaining_nulls = df[remaining_numeric].isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"  Filling {remaining_nulls} remaining NaNs with column medians")
        df[remaining_numeric] = df[remaining_numeric].fillna(df[remaining_numeric].median())

    # 4. Fill categorical NaNs with 'Unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna("Unknown")

    out_path = os.path.join(INTERIM_DIR, "03_preprocessed.csv")
    df.to_csv(out_path, index=False)
    print(f"  ✓ Preprocessing complete → {df.shape[0]} rows, {df.shape[1]} cols")


if __name__ == "__main__":
    preprocess()
