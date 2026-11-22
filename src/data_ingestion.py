from pathlib import Path

import pandas as pd
import yaml

OUTPUT_PATH = Path("data/ingested/bank.csv")

# Columns present in both the UCI bank-full and bank-additional variants.
# Variant-specific columns (balance/day vs. economic indicators) are kept as-is.
CORE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    cfg = params["data"]
    raw_path = Path(cfg["raw_path"])
    target = cfg["target"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {raw_path}. Put the Bank Marketing CSV at this location."
        )

    data = pd.read_csv(raw_path, sep=cfg.get("separator", ","))
    data.columns = data.columns.str.strip().str.strip('"')

    missing_columns = sorted(set(CORE_COLUMNS + [target]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("The input dataset is empty.")

    # Trim stray whitespace in text columns (levels like ' yes' vs 'yes').
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].str.strip()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Data ingestion completed: {data.shape[0]} rows, {data.shape[1]} columns")
    print("Target distribution:")
    print(data[target].value_counts(normalize=True).round(4).to_string())
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
