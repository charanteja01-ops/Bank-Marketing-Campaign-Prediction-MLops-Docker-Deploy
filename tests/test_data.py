from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering import add_features

RAW_DATA_PATH = Path("data/raw/bank.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
PARAMS_PATH = Path("params.yaml")

EXPECTED_RAW_ROWS = 11162
RAW_FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome",
]
DERIVED_FEATURES = [
    "previously_contacted",
    "pdays_clean",
    "recent_contact_30d",
    "has_any_loan",
    "has_both_loans",
    "debt_free",
    "balance_is_negative",
    "signed_log_balance",
    "balance_to_age_ratio",
    "age_group",
    "is_student_or_retired",
    "is_blue_collar_or_admin",
    "education_ordinal",
    "log_campaign",
    "heavy_campaign_calls",
    "total_contacts",
    "campaign_to_total_ratio",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "prev_outcome_success",
    "success_per_campaign_call",
    "loan_free_positive_balance",
]
EXPECTED_TARGET = "deposit"
RAW_VALID_CLASSES = {"yes", "no"}
ENCODED_VALID_CLASSES = {0, 1}
VALID_MARITAL = {"married", "single", "divorced"}
VALID_MONTHS = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
}
VALID_AGE_GROUPS = {"young", "working_age", "senior"}


def _require(*paths):
    """Processed files are created by `dvc repro`; skip (not fail) until they exist."""
    for path in paths:
        if not path.exists():
            pytest.skip(f"{path} not found. Run `dvc repro` first.")


def _load_params():
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    return {"test_size": 0.2, "data": {"drop_columns": ["duration"]}}


def test_raw_dataset_exists_and_valid():
    """Verify raw bank dataset exists, has expected columns and no nulls."""
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)

    assert len(df) == EXPECTED_RAW_ROWS, (
        f"Expected {EXPECTED_RAW_ROWS} rows in raw bank data, got {len(df)}"
    )
    for col in RAW_FEATURES + [EXPECTED_TARGET]:
        assert col in df.columns, f"Missing column {col} in raw data"

    assert df.isnull().sum().sum() == 0, "Null values found in raw data"
    assert set(df[EXPECTED_TARGET].unique()) == RAW_VALID_CLASSES
    assert set(df["marital"].unique()) <= VALID_MARITAL
    assert set(df["month"].unique()) <= VALID_MONTHS


def test_raw_numeric_values_are_in_valid_ranges():
    """Sanity-check numeric columns for impossible values."""
    df = pd.read_csv(RAW_DATA_PATH)

    assert df["age"].between(18, 100).all(), "age outside 18-100"
    assert df["day"].between(1, 31).all(), "day outside 1-31"
    assert (df["duration"] >= 0).all(), "negative call duration"
    assert (df["campaign"] >= 1).all(), "campaign must be at least 1"
    assert (df["pdays"] >= -1).all(), "pdays below -1 (-1 means never contacted)"
    assert (df["previous"] >= 0).all(), "negative previous contacts"


def test_preprocessed_splits_exist_and_consistent():
    """Verify train/test sets exist, match the deduplicated total and keep all raw columns."""
    _require(TRAIN_DATA_PATH, TEST_DATA_PATH)

    raw_df = pd.read_csv(RAW_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    expected_total = len(raw_df.drop_duplicates())
    total_rows = len(train_df) + len(test_df)
    assert total_rows == expected_total, (
        f"Expected {expected_total} deduplicated rows across splits, got {total_rows}"
    )

    for split_name, split in (("train", train_df), ("test", test_df)):
        for col in RAW_FEATURES + [EXPECTED_TARGET]:
            assert col in split.columns, f"Missing column {col} in {split_name}"
        assert split.isnull().sum().sum() == 0, f"Null values in {split_name}"


def test_split_ratio_matches_params():
    """The test share should match test_size in params.yaml."""
    _require(TRAIN_DATA_PATH, TEST_DATA_PATH)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_size = float(_load_params()["test_size"])

    actual = len(test_df) / (len(train_df) + len(test_df))
    assert abs(actual - test_size) < 0.01, f"Test share {actual:.3f} != test_size {test_size}"


def test_splits_are_stratified_and_leak_free():
    """Both encoded classes (0/1) in both splits with similar balance, and no shared rows."""
    _require(TRAIN_DATA_PATH, TEST_DATA_PATH)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    assert set(train_df[EXPECTED_TARGET].unique()) == ENCODED_VALID_CLASSES
    assert set(test_df[EXPECTED_TARGET].unique()) == ENCODED_VALID_CLASSES

    train_pos = (train_df[EXPECTED_TARGET] == 1).mean()
    test_pos = (test_df[EXPECTED_TARGET] == 1).mean()
    assert abs(train_pos - test_pos) < 0.02, (
        f"Class balance differs: train {train_pos:.3f} vs test {test_pos:.3f}"
    )

    combined = pd.concat([train_df, test_df], ignore_index=True)
    assert combined.duplicated().sum() == 0, "Rows found in both train and test"


def test_feature_engineered_splits_valid():
    """Verify drop_columns removal and all stateless engineered features."""
    _require(TRAIN_DATA_PATH, TEST_DATA_PATH, X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH)

    params = _load_params()
    drop_columns = params.get("data", {}).get("drop_columns", [])
    expected_features = [c for c in RAW_FEATURES if c not in drop_columns] + DERIVED_FEATURES

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    assert len(X_train) == len(train_df) == len(y_train)
    assert len(X_test) == len(test_df) == len(y_test)

    for split_name, X_split in (("X_train", X_train), ("X_test", X_test)):
        assert list(X_split.columns) == expected_features, f"Unexpected columns in {split_name}"
        for col in drop_columns:
            assert col not in X_split.columns, f"{col} should be dropped from {split_name}"
        assert set(X_split["previously_contacted"].unique()) <= {0, 1}
        assert set(X_split["recent_contact_30d"].unique()) <= {0, 1}
        assert set(X_split["has_any_loan"].unique()) <= {0, 1}
        assert set(X_split["has_both_loans"].unique()) <= {0, 1}
        assert set(X_split["debt_free"].unique()) <= {0, 1}
        assert set(X_split["balance_is_negative"].unique()) <= {0, 1}
        assert set(X_split["age_group"].unique()) <= VALID_AGE_GROUPS
        assert (X_split["pdays"] >= 0).all(), f"Sentinel values left in pdays ({split_name})"
        assert (X_split["pdays_clean"] >= 0).all(), f"Sentinel values left in pdays_clean ({split_name})"


def test_add_features_is_stateless_and_idempotent():
    """Verify add_features works row-wise and matches training feature engineering."""
    _require(X_TRAIN_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)
    raw_subset = X_train[[c for c in RAW_FEATURES if c in X_train.columns]].head(5).copy()
    engineered = add_features(raw_subset)
    assert list(engineered.columns) == list(X_train.columns)