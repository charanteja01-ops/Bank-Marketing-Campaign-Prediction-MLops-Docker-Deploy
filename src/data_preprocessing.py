from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/ingested/bank.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


def clean_strings(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col != target:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    return df


def cap_outliers_iqr(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    factor: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute bounds strictly from train_df and clip both train and test."""
    train_out = train_df.copy()
    test_out = test_df.copy()

    for col in cols:
        if col not in train_out.columns or col not in test_out.columns:
            continue
        s_train = pd.to_numeric(train_out[col], errors="coerce")
        q1 = s_train.quantile(0.25)
        q3 = s_train.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            low = q1 - factor * iqr
            high = q3 + factor * iqr
            train_out[col] = s_train.clip(lower=low, upper=high)
            test_out[col] = pd.to_numeric(test_out[col], errors="coerce").clip(lower=low, upper=high)

    return train_out, test_out


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    target = params["data"]["target"]
    positive_label = str(params["data"]["positive_label"]).strip().lower()

    data = pd.read_csv(INPUT_PATH)

    # 1. Remove exact duplicate records before splitting.
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(data)} duplicate rows.")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    # 2. Standardize string values cleanly.
    data = clean_strings(data, target=target)

    # 3. Encode target as 1 (positive_label) and 0 otherwise.
    labels = data[target].astype(str).str.strip().str.lower()
    if labels.nunique() != 2:
        raise ValueError(
            f"Expected a binary target, found values: {sorted(labels.unique())}"
        )
    if positive_label not in set(labels):
        raise ValueError(f"positive_label '{positive_label}' not found in target.")
    data[target] = (labels == positive_label).astype(int)

    X = data.drop(columns=[target])
    y = data[target]

    # 4. Stratified split first so any bound computation uses training rows only.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
        stratify=y,
    )

    outlier_cols = [c for c in ["balance", "campaign", "previous"] if c in X_train.columns]
    X_train, X_test = cap_outliers_iqr(X_train, X_test, outlier_cols, factor=2.5)

    train = X_train.copy()
    train[target] = y_train
    test = X_test.copy()
    test[target] = y_test

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Training data: {train.shape}  positive rate: {y_train.mean():.4f}")
    print(f"Testing data: {test.shape}  positive rate: {y_test.mean():.4f}")
    print(f"Saved: {TRAIN_PATH} and {TEST_PATH}")


if __name__ == "__main__":
    main()