"""
Unit Tests: Data Validation (Task 5 — CI/CD)
Tests run during CI to verify data integrity.
"""
import pandas as pd
import os
import pytest

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Data", "interim", "04_engineered.csv"
)


@pytest.fixture
def sample_data():
    """Load sample data if available, otherwise create mock data."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, nrows=500)
    else:
        # Mock data for CI environment where real data may not exist
        return pd.DataFrame({
            "isFraud": [0, 1, 0, 0, 1],
            "TransactionAmt": [100.0, 250.0, 50.0, 75.0, 500.0],
            "TransactionDT": [100, 200, 300, 400, 500],
            "card1": [1000, 2000, 3000, 4000, 5000],
        })


def test_target_column_exists(sample_data):
    """Target column isFraud must exist."""
    assert "isFraud" in sample_data.columns, "Target column 'isFraud' is missing"


def test_target_values_binary(sample_data):
    """Target column must contain only 0 and 1."""
    valid_values = {0, 1}
    actual_values = set(sample_data["isFraud"].dropna().unique())
    assert actual_values.issubset(valid_values), f"Invalid target values: {actual_values}"


def test_no_negative_amounts(sample_data):
    """Transaction amounts should not be negative."""
    if "TransactionAmt" in sample_data.columns:
        assert (sample_data["TransactionAmt"] >= 0).all(), "Negative transaction amounts found"


def test_schema_minimum_columns(sample_data):
    """Dataset must have minimum number of columns."""
    assert sample_data.shape[1] >= 3, f"Too few columns: {sample_data.shape[1]}"


def test_no_empty_dataframe(sample_data):
    """Dataset must not be empty."""
    assert len(sample_data) > 0, "DataFrame is empty"


def test_target_not_all_null(sample_data):
    """Target column must not be entirely null."""
    assert not sample_data["isFraud"].isnull().all(), "Target column is entirely null"
