from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TRAIN_INPUT = Path("data/processed/train.csv")
TEST_INPUT = Path("data/processed/test.csv")

X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
EDUCATION_RANK = {
    "unknown": 0,
    "illiterate": 0,
    "primary": 1,
    "secondary": 2,
    "tertiary": 3,
}


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, stateless domain and interaction features (zero data leakage)."""
    X = X.copy()

    # 1. Contact history & pdays sentinel cleanup
    if "pdays" in X.columns:
        pdays_num = pd.to_numeric(X["pdays"], errors="coerce").fillna(-1)
        contacted = ~pdays_num.isin([-1, 999])
        X["previously_contacted"] = contacted.astype(int)
        X["pdays"] = pd.to_numeric(X["pdays"], errors="coerce").where(contacted, 0)
        X["pdays_clean"] = pd.to_numeric(X["pdays"], errors="coerce").where(contacted, 0)
        X["recent_contact_30d"] = (contacted & (X["pdays_clean"] <= 30)).astype(int)

    # 2. Loan & debt indicators
    has_housing = (
        X["housing"].astype(str).str.strip().str.lower().eq("yes")
        if "housing" in X.columns
        else pd.Series(False, index=X.index)
    )
    has_personal = (
        X["loan"].astype(str).str.strip().str.lower().eq("yes")
        if "loan" in X.columns
        else pd.Series(False, index=X.index)
    )
    if "housing" in X.columns or "loan" in X.columns:
        X["has_any_loan"] = (has_housing | has_personal).astype(int)
        X["has_both_loans"] = (has_housing & has_personal).astype(int)
        X["debt_free"] = ((~has_housing) & (~has_personal)).astype(int)

    # 3. Account balance transforms & age ratios
    if "balance" in X.columns:
        bal = pd.to_numeric(X["balance"], errors="coerce").fillna(0.0)
        X["balance_is_negative"] = (bal < 0).astype(int)
        X["signed_log_balance"] = np.sign(bal) * np.log1p(np.abs(bal))

        if "age" in X.columns:
            age = pd.to_numeric(X["age"], errors="coerce").fillna(40.0)
            X["balance_to_age_ratio"] = (bal / (age.clip(lower=18.0) + 1.0)).round(4)

    # 4. Age grouping & low-susceptibility demographic flag
    if "age" in X.columns:
        age = pd.to_numeric(X["age"], errors="coerce").fillna(40.0)
        X["age_group"] = pd.cut(
            age,
            bins=[0, 29, 59, 120],
            labels=["young", "working_age", "senior"],
            include_lowest=True,
        ).astype(str)

    if "job" in X.columns:
        job_clean = X["job"].astype(str).str.strip().str.lower()
        X["is_student_or_retired"] = job_clean.isin(["student", "retired"]).astype(int)
        X["is_blue_collar_or_admin"] = job_clean.isin(["blue-collar", "admin."]).astype(int)

    # 5. Ordinal education & campaign effort ratios
    if "education" in X.columns:
        edu_clean = X["education"].astype(str).str.strip().str.lower()
        X["education_ordinal"] = edu_clean.map(EDUCATION_RANK).fillna(0).astype(int)

    if "campaign" in X.columns:
        camp = pd.to_numeric(X["campaign"], errors="coerce").fillna(1.0).clip(lower=1.0)
        X["log_campaign"] = np.log1p(camp)
        X["heavy_campaign_calls"] = (camp >= 4).astype(int)

        if "previous" in X.columns:
            prev = pd.to_numeric(X["previous"], errors="coerce").fillna(0.0).clip(lower=0.0)
            X["total_contacts"] = camp + prev
            X["campaign_to_total_ratio"] = (camp / (camp + prev + 1.0)).round(4)

    # 6. Cyclical month & day encoding
    if "month" in X.columns:
        month_num = (
            X["month"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str[:3]
            .map(MONTH_MAP)
            .fillna(5)
        )
        X["month_sin"] = np.sin(2.0 * np.pi * month_num / 12.0).round(6)
        X["month_cos"] = np.cos(2.0 * np.pi * month_num / 12.0).round(6)

    if "day" in X.columns:
        day_num = pd.to_numeric(X["day"], errors="coerce").fillna(15).clip(1, 31)
        X["day_sin"] = np.sin(2.0 * np.pi * day_num / 31.0).round(6)
        X["day_cos"] = np.cos(2.0 * np.pi * day_num / 31.0).round(6)

    # 7. High-signal interaction features
    if "poutcome" in X.columns:
        pout = X["poutcome"].astype(str).str.strip().str.lower()
        X["prev_outcome_success"] = pout.eq("success").astype(int)
        if "campaign" in X.columns:
            camp = pd.to_numeric(X["campaign"], errors="coerce").fillna(1.0)
            X["success_per_campaign_call"] = (
                X["prev_outcome_success"] / (camp.clip(lower=1.0))
            ).round(4)

    if "balance" in X.columns and "has_any_loan" in X.columns:
        bal = pd.to_numeric(X["balance"], errors="coerce").fillna(0.0)
        X["loan_free_positive_balance"] = (
            (X["has_any_loan"] == 0) & (bal > 0)
        ).astype(int)

    return X


def transform_split(input_path: Path, target: str, drop_columns: list[str]):
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
    print(f"Features ({len(X_train.columns)}): {list(X_train.columns)}")
    print(f"Saved split files under: {X_TRAIN_PATH.parent}")


if __name__ == "__main__":
    main()