from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering import add_features
from src.monitor_drift import (
    FEATURE_COLUMNS,
    analyze_drift,
    generate_alert_markdown,
    log_inference,
    send_drift_alert,
)


def _build_bank_dataframe(n_rows=150, seed=42, shifted=False):
    """Generate realistic Bank Marketing records matching FEATURE_COLUMNS."""
    rng = np.random.RandomState(seed)

    if not shifted:
        age = rng.normal(41.0, 10.0, n_rows).clip(18, 95).round().astype(int)
        balance = rng.normal(1400.0, 800.0, n_rows).round(2)
        housing = rng.choice(["yes", "no"], n_rows, p=[0.55, 0.45])
        loan = rng.choice(["no", "yes"], n_rows, p=[0.84, 0.16])
        raw_data = {
            "age": age,
            "job": rng.choice(
                ["management", "blue-collar", "technician", "admin."],
                n_rows,
                p=[0.35, 0.30, 0.20, 0.15],
            ),
            "marital": rng.choice(
                ["married", "single", "divorced"], n_rows, p=[0.60, 0.28, 0.12]
            ),
            "education": rng.choice(
                ["secondary", "tertiary", "primary"], n_rows, p=[0.50, 0.35, 0.15]
            ),
            "default": rng.choice(["no", "yes"], n_rows, p=[0.98, 0.02]),
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": rng.choice(
                ["cellular", "unknown", "telephone"], n_rows, p=[0.65, 0.28, 0.07]
            ),
            "day": rng.randint(1, 29, n_rows),
            "month": rng.choice(
                ["may", "jul", "aug", "jun", "nov"],
                n_rows,
                p=[0.35, 0.20, 0.20, 0.15, 0.10],
            ),
            "duration": rng.normal(250.0, 80.0, n_rows).clip(10, 1200).round().astype(int),
            "campaign": rng.poisson(2.0, n_rows) + 1,
            "pdays": np.zeros(n_rows, dtype=int),
            "previous": np.zeros(n_rows, dtype=int),
            "poutcome": rng.choice(
                ["unknown", "failure", "other", "success"],
                n_rows,
                p=[0.80, 0.11, 0.05, 0.04],
            ),
        }
    else:
        age = rng.normal(72.0, 4.0, n_rows).clip(60, 95).round().astype(int)
        balance = rng.normal(-450.0, 150.0, n_rows).round(2)
        housing = rng.choice(["no", "yes"], n_rows, p=[0.95, 0.05])
        loan = rng.choice(["no", "yes"], n_rows, p=[0.98, 0.02])
        raw_data = {
            "age": age,
            "job": rng.choice(["retired", "student"], n_rows, p=[0.85, 0.15]),
            "marital": rng.choice(["divorced", "single"], n_rows, p=[0.80, 0.20]),
            "education": rng.choice(["primary", "unknown"], n_rows, p=[0.85, 0.15]),
            "default": rng.choice(["yes", "no"], n_rows, p=[0.75, 0.25]),
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": rng.choice(["telephone", "unknown"], n_rows, p=[0.85, 0.15]),
            "day": rng.randint(25, 32, n_rows),
            "month": rng.choice(["dec", "mar", "oct"], n_rows, p=[0.50, 0.30, 0.20]),
            "duration": rng.normal(950.0, 100.0, n_rows).clip(600, 2000).round().astype(int),
            "campaign": rng.normal(18.0, 2.0, n_rows).clip(10, 40).round().astype(int),
            "pdays": rng.normal(300.0, 25.0, n_rows).clip(180, 500).round().astype(int),
            "previous": rng.normal(10.0, 1.5, n_rows).clip(5, 20).round().astype(int),
            "poutcome": rng.choice(["success", "other"], n_rows, p=[0.85, 0.15]),
        }

    df = add_features(pd.DataFrame(raw_data))
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[available]


def test_log_inference_writes_record(tmp_path):
    """Verify that logging inference appends a valid structured record."""
    test_log = tmp_path / "test_logs.csv"
    input_sample = _build_bank_dataframe(n_rows=1, seed=42, shifted=False)

    log_inference(
        input_sample,
        "yes",
        {"no": 0.18, "yes": 0.82},
        log_path=test_log,
    )

    assert test_log.exists()
    logged_df = pd.read_csv(test_log)
    assert len(logged_df) == 1
    assert str(logged_df.iloc[0]["predicted_class"]) == "yes"
    assert "timestamp" in logged_df.columns
    assert np.isclose(logged_df.iloc[0]["prob_yes"], 0.82)


def test_analyze_drift_no_drift_when_identical():
    """Verify that identical distributions report no drift."""
    ref_df = _build_bank_dataframe(n_rows=150, seed=42, shifted=False)
    curr_df = ref_df.copy()

    report = analyze_drift(ref_df, curr_df)
    assert report["dataset_drift_detected"] is False
    assert report["drifted_features"] == 0
    assert report["drift_share_pct"] == 0.0


def test_analyze_drift_detects_distribution_shift():
    """Verify that significant numeric and categorical shifts are detected."""
    ref_df = _build_bank_dataframe(n_rows=150, seed=42, shifted=False)
    curr_df = _build_bank_dataframe(n_rows=150, seed=99, shifted=True)

    report = analyze_drift(ref_df, curr_df)
    assert report["dataset_drift_detected"] is True
    assert report["drifted_features"] >= 1


def test_analyze_drift_detects_categorical_shift():
    """Verify Chi-Square test detects distribution shift in categorical features."""
    ref_df = _build_bank_dataframe(n_rows=150, seed=42, shifted=False)
    curr_df = ref_df.copy()

    if "job" in curr_df.columns:
        curr_df["job"] = "retired"
    if "month" in curr_df.columns:
        curr_df["month"] = "dec"
    if "poutcome" in curr_df.columns:
        curr_df["poutcome"] = "success"

    report = analyze_drift(ref_df, curr_df)
    assert report["drifted_features"] >= 1


def test_generate_alert_markdown_creates_valid_report(tmp_path):
    """Verify that alert markdown is written with all tables and headers."""
    alert_path = tmp_path / "test_alert.md"
    ref_df = _build_bank_dataframe(n_rows=150, seed=42, shifted=False)
    curr_df = _build_bank_dataframe(n_rows=150, seed=99, shifted=True)

    report = analyze_drift(ref_df, curr_df)
    md_text = generate_alert_markdown(report, output_path=alert_path)

    assert alert_path.exists()
    assert "DATA DRIFT DETECTED" in md_text
    assert "Continuous Training Required" in md_text
    assert "KS-test" in md_text or "chi-square" in md_text.lower()


def test_send_drift_alert_graceful_without_webhook():
    """Verify send_drift_alert handles missing webhook without crashing."""
    total = len(FEATURE_COLUMNS)
    report = {
        "dataset_drift_detected": True,
        "drifted_features": total,
        "total_features": total,
        "drift_share_pct": 100.0,
    }
    result = send_drift_alert(report, webhook_url=None)
    assert result is False