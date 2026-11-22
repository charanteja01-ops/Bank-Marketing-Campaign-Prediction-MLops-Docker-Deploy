from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/ingested/bank.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    target = params["data"]["target"]
    positive_label = str(params["data"]["positive_label"]).lower()

    data = pd.read_csv(INPUT_PATH)

    # Remove exact duplicate records before splitting.
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(data)} duplicate rows.")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    # Encode the target as 1 (subscribed / yes) and 0 (no).
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
        stratify=y,
    )

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
