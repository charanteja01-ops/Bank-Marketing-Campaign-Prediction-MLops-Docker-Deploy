from pathlib import Path

import pandas as pd
import yaml

TRAIN_INPUT = Path("data/processed/train.csv")
TEST_INPUT = Path("data/processed/test.csv")

X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")


def add_features(X):
    """Row-wise, stateless transformations only (nothing is learned from the
    data), so applying them to train and test separately cannot leak.

    pdays is -1 (bank-full) or 999 (bank-additional) when the client was never
    contacted before. That sentinel would distort scaling, so we expose it as a
    clean flag and set pdays to 0 for those rows.
    """
    X = X.copy()
    if "pdays" in X.columns:
        contacted = ~X["pdays"].isin([-1, 999])
        X["previously_contacted"] = contacted.astype(int)
        X["pdays"] = X["pdays"].where(contacted, 0)
    return X


def transform_split(input_path, target, drop_columns):
    data = pd.read_csv(input_path)

    if target not in data.columns:
        raise ValueError(f"Missing target '{target}' in {input_path}")

    y = data[target].copy()
    X = data.drop(columns=[target])
    X = X.drop(columns=[c for c in drop_columns if c in X.columns])
    X = add_features(X)

    # Scaling and one-hot encoding stay inside the sklearn Pipeline so they
    # are fitted on training folds only.
    return X, y


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    target = params["data"]["target"]
    drop_columns = params["data"].get("drop_columns", [])

    X_train, y_train = transform_split(TRAIN_INPUT, target, drop_columns)
    X_test, y_test = transform_split(TEST_INPUT, target, drop_columns)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns do not match.")

    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame(name=target).to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame(name=target).to_csv(Y_TEST_PATH, index=False)

    print("Feature engineering and target separation completed.")
    print(f"Dropped columns: {drop_columns}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    print(f"Saved split files under: {X_TRAIN_PATH.parent}")


if __name__ == "__main__":
    main()
