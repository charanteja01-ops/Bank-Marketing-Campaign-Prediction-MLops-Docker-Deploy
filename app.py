"""Streamlit MLOps serving application for Bank Marketing Campaign Prediction.

Features:
- Live single-customer and batch CSV inference with probability visualization.
- Automatic production payload & prediction logging for data drift tracking.
- Interactive champion model metadata, multi-criteria benchmark & lineage inspector.
- Real-time statistical data drift monitoring dashboard (KS-Test + Chi-Square).
"""
import io
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SELECTION_PATH = BASE_DIR / "reports" / "model_selection.json"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
DETAILS_PATH = BASE_DIR / "reports" / "evaluation_details.json"
COMPARISON_CSV_PATH = BASE_DIR / "reports" / "model_comparison.csv"

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

try:
    from src.monitor_drift import (
        FEATURE_COLUMNS,
        INFERENCE_LOG_PATH,
        REFERENCE_DATA_PATH,
        analyze_drift,
        generate_sample_production_data,
        load_inference_logs,
        log_inference,
    )
except ImportError:
    try:
        from monitor_drift import (  # type: ignore[import-not-found]
            FEATURE_COLUMNS,
            INFERENCE_LOG_PATH,
            REFERENCE_DATA_PATH,
            analyze_drift,
            generate_sample_production_data,
            load_inference_logs,
            log_inference,
        )
    except ImportError:
        INFERENCE_LOG_PATH = BASE_DIR / "reports" / "inference_logs.csv"
        REFERENCE_DATA_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"
        FEATURE_COLUMNS = [
            "age", "job", "marital", "education", "default", "balance",
            "housing", "loan", "contact", "day", "month", "campaign",
            "pdays", "previous", "poutcome", "previously_contacted",
            "pdays_clean", "recent_contact_30d", "has_any_loan",
            "has_both_loans", "debt_free", "balance_is_negative",
            "signed_log_balance", "balance_to_age_ratio", "age_group",
            "is_student_or_retired", "is_blue_collar_or_admin",
            "education_ordinal", "log_campaign", "heavy_campaign_calls",
            "total_contacts", "campaign_to_total_ratio", "month_sin",
            "month_cos", "day_sin", "day_cos", "prev_outcome_success",
            "success_per_campaign_call", "loan_free_positive_balance",
        ]
        _NUMERIC_FEATURES = {
            "age", "balance", "day", "campaign", "pdays", "previous",
            "pdays_clean", "signed_log_balance", "balance_to_age_ratio",
            "education_ordinal", "log_campaign", "total_contacts",
            "campaign_to_total_ratio", "month_sin", "month_cos",
            "day_sin", "day_cos", "success_per_campaign_call",
        }

        def _ks_2samp(x, y):
            x = np.sort(np.asarray(x, dtype=float))
            y = np.sort(np.asarray(y, dtype=float))
            n1, n2 = len(x), len(y)
            combined = np.concatenate([x, y])
            cdf1 = np.searchsorted(x, combined, side="right") / n1
            cdf2 = np.searchsorted(y, combined, side="right") / n2
            d = float(np.max(np.abs(cdf1 - cdf2)))
            en = math.sqrt(n1 * n2 / (n1 + n2))
            lam = (en + 0.12 + 0.11 / en) * d
            if lam <= 0.0:
                return d, 1.0
            j = np.arange(1, 101)
            p = float(np.clip(2.0 * np.sum(((-1.0) ** (j - 1)) * np.exp(-2.0 * (j ** 2) * (lam ** 2))), 0.0, 1.0))
            return d, p

        def _gammaincc(a, x):
            if x <= 0.0:
                return 1.0
            if x < a + 1.0:
                ap = a
                total = 1.0 / a
                delta = total
                for _ in range(200):
                    ap += 1.0
                    delta *= x / ap
                    total += delta
                    if abs(delta) < abs(total) * 1e-14:
                        break
                p_lower = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
                return max(0.0, min(1.0, 1.0 - p_lower))
            b = x + 1.0 - a
            c = 1.0 / 1e-30
            d = 1.0 / b
            h = d
            for i in range(1, 201):
                an = -i * (i - a)
                b += 2.0
                d = an * d + b
                if abs(d) < 1e-30:
                    d = 1e-30
                c = b + an / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < 1e-14:
                    break
            return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h))

        def _chi2_contingency(contingency):
            obs = np.asarray(contingency, dtype=float)
            total = obs.sum()
            if total <= 0:
                return 0.0, 1.0
            expected = (obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True)) / total
            mask = expected > 0
            chi2_stat = float(np.sum(((obs[mask] - expected[mask]) ** 2) / expected[mask]))
            dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
            p_val = _gammaincc(dof / 2.0, chi2_stat / 2.0) if dof > 0 else 1.0
            return chi2_stat, float(p_val)

        def load_inference_logs(log_path=INFERENCE_LOG_PATH):
            path = Path(log_path)
            return pd.read_csv(path) if path.exists() else pd.DataFrame()

        def log_inference(input_df, prediction, proba_dict=None, log_path=INFERENCE_LOG_PATH):
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            cols = [c for c in FEATURE_COLUMNS if c in input_df.columns]
            record = input_df[cols].copy() if cols else input_df.copy()
            record["predicted_class"] = prediction
            if proba_dict:
                for k, v in proba_dict.items():
                    record[f"prob_{k}"] = float(v)
            record["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
            record.to_csv(path, mode="a", header=not path.exists(), index=False)

        def generate_sample_production_data(ref_df, simulate_drift=False, n_samples=100, seed=42):
            return ref_df.sample(n=min(n_samples, len(ref_df)), replace=True, random_state=seed).reset_index(drop=True)

        def analyze_drift(ref_df, curr_df, p_value_threshold=0.05, drift_share_threshold=0.50):
            available = [c for c in FEATURE_COLUMNS if c in ref_df.columns and c in curr_df.columns]
            if not available:
                raise ValueError("No overlapping feature columns found between reference and current data.")
            feature_results = {}
            drifted_count = 0
            for col in available:
                ref_s = ref_df[col].dropna()
                cur_s = curr_df[col].dropna()
                if len(ref_s) == 0 or len(cur_s) == 0:
                    continue
                if col in _NUMERIC_FEATURES:
                    r_v = pd.to_numeric(ref_s, errors="coerce").dropna()
                    c_v = pd.to_numeric(cur_s, errors="coerce").dropna()
                    stat, p_val = _ks_2samp(r_v, c_v)
                    drifted = bool(p_val < p_value_threshold)
                    r_m, c_m = float(r_v.mean()), float(c_v.mean())
                    feature_results[col] = {
                        "test_name": "KS-test",
                        "drift_detected": drifted,
                        "p_value": float(p_val),
                        "statistic": float(stat),
                        "reference_mean": r_m,
                        "current_mean": c_m,
                        "mean_shift": float(c_m - r_m),
                    }
                else:
                    rc = ref_s.astype(str).value_counts()
                    cc = cur_s.astype(str).value_counts()
                    cats = sorted(set(rc.index) | set(cc.index))
                    if len(cats) < 2:
                        stat, p_val = 0.0, 1.0
                    else:
                        cont = np.array([[rc.get(k, 0) for k in cats], [cc.get(k, 0) for k in cats]])
                        stat, p_val = _chi2_contingency(cont)
                    drifted = bool(p_val < p_value_threshold)
                    feature_results[col] = {
                        "test_name": "Chi-square",
                        "drift_detected": drifted,
                        "p_value": float(p_val),
                        "statistic": float(stat),
                        "reference_mean": None,
                        "current_mean": None,
                        "mean_shift": None,
                    }
                if drifted:
                    drifted_count += 1
            total = len(feature_results)
            share = round((drifted_count / total) * 100.0, 2) if total > 0 else 0.0
            return {
                "dataset_drift_detected": bool((drifted_count / total) >= drift_share_threshold) if total > 0 else False,
                "drifted_features": drifted_count,
                "total_features": total,
                "drift_share_pct": share,
                "features": feature_results,
            }

TARGET = "deposit"
POSITIVE_LABEL = "yes"

RAW_INPUTS = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "campaign", "pdays", "previous", "poutcome",
]
DERIVED = [
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
ALLOWED_COLUMNS = set(RAW_INPUTS) | set(DERIVED)

# High-value inputs shown in the form (lower-priority/noisy fields like day,
# default, pdays, and previous are omitted from the UI and defaulted cleanly).
CORE_FORM_INPUTS = [
    "age", "job", "marital", "education",
    "balance", "housing", "loan",
    "contact", "month", "campaign", "poutcome",
]

OPTIONS = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ],
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["yes", "no"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ],
    "poutcome": ["unknown", "failure", "success", "other"],
}

DEFAULT_VALUES = {
    "age": 41, "job": "management", "marital": "married", "education": "secondary",
    "default": "no", "balance": 1500, "housing": "yes", "loan": "no",
    "contact": "cellular", "day": 15, "month": "may", "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown",
    "previously_contacted": 0, "pdays_clean": 0, "recent_contact_30d": 0,
    "has_any_loan": 1, "has_both_loans": 0, "debt_free": 0,
    "balance_is_negative": 0, "signed_log_balance": round(float(np.log1p(1500)), 4),
    "balance_to_age_ratio": round(1500 / 42.0, 4), "age_group": "working_age",
    "is_student_or_retired": 0, "is_blue_collar_or_admin": 0,
    "education_ordinal": 2, "log_campaign": round(float(np.log1p(2)), 4),
    "heavy_campaign_calls": 0, "total_contacts": 2,
    "campaign_to_total_ratio": round(2 / 3.0, 4),
    "month_sin": round(float(np.sin(2.0 * np.pi * 5 / 12.0)), 6),
    "month_cos": round(float(np.cos(2.0 * np.pi * 5 / 12.0)), 6),
    "day_sin": round(float(np.sin(2.0 * np.pi * 15 / 31.0)), 6),
    "day_cos": round(float(np.cos(2.0 * np.pi * 15 / 31.0)), 6),
    "prev_outcome_success": 0, "success_per_campaign_call": 0.0,
    "loan_free_positive_balance": 0,
}

PRESETS = {
    "Pick an example profile...": None,
    "Example A: Young professional (high intent)": {
        "age": 34, "job": "management", "marital": "single", "education": "tertiary",
        "balance": 3500, "housing": "no", "loan": "no",
        "contact": "cellular", "month": "mar", "campaign": 1, "poutcome": "success",
    },
    "Example B: Retired saver": {
        "age": 67, "job": "retired", "marital": "married", "education": "secondary",
        "balance": 5200, "housing": "no", "loan": "no",
        "contact": "cellular", "month": "oct", "campaign": 1, "poutcome": "unknown",
    },
    "Example C: Blue-collar with loans": {
        "age": 41, "job": "blue-collar", "marital": "married", "education": "primary",
        "balance": 250, "housing": "yes", "loan": "yes",
        "contact": "unknown", "month": "may", "campaign": 3, "poutcome": "unknown",
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.block-container {
    padding-top: 1.35rem;
    padding-bottom: 2.5rem;
    max-width: 1220px;
}

.hero {
    background: linear-gradient(125deg, #0f172a 0%, #1d4ed8 48%, #0d9488 85%, #f59e0b 125%);
    color: #ffffff;
    padding: 1.65rem 2.1rem;
    border-radius: 22px;
    margin-bottom: 1.35rem;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.22);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

.hero-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(6px);
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
    color: #fef08a;
}

.hero h1 {
    margin: 0;
    font-size: 2.05rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
}

.hero p {
    margin: 0.45rem 0 0 0;
    opacity: 0.94;
    font-size: 1.02rem;
    color: #f1f5f9;
}

.group-card {
    border-radius: 16px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.75rem;
    font-weight: 700;
    font-size: 0.96rem;
    border-left: 5px solid;
}

.group-profile {
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.14), rgba(37, 99, 235, 0.03));
    border-left-color: #2563eb;
    color: #1e40af;
}

.group-finance {
    background: linear-gradient(90deg, rgba(13, 148, 136, 0.16), rgba(13, 148, 136, 0.03));
    border-left-color: #0d9488;
    color: #0f766e;
}

.group-campaign {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.18), rgba(245, 158, 11, 0.03));
    border-left-color: #d97706;
    color: #b45309;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(59, 130, 246, 0.22);
    border-radius: 16px;
    padding: 0.8rem 1rem;
    background: linear-gradient(145deg, rgba(37, 99, 235, 0.07), rgba(13, 148, 136, 0.05));
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}

button[data-baseweb="tab"] {
    font-weight: 700 !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0.55rem 1rem !important;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #0d9488 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.28) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 22px rgba(13, 148, 136, 0.34) !important;
}

.ring-wrap {
    position: relative;
    width: 184px;
    height: 184px;
    margin: 0 auto;
}

.ring {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(148, 163, 184, 0.22) 0);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 18px), #000 calc(100% - 17px));
            mask: radial-gradient(farthest-side, transparent calc(100% - 18px), #000 calc(100% - 17px));
}

.ring-text {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.ring-text .pct {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
}

.ring-text .lbl {
    font-size: 0.78rem;
    opacity: 0.75;
    margin-top: 0.28rem;
    font-weight: 600;
}

.badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    color: #ffffff;
    font-size: 0.95rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

.small-note {
    font-size: 0.85rem;
    opacity: 0.8;
}
</style>
"""


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Idempotent stateless feature engineering matching src/feature_engineering.py."""
    X = X.copy()
    for col, val in DEFAULT_VALUES.items():
        if col in RAW_INPUTS and col not in X.columns:
            X[col] = val

    if "pdays" in X.columns:
        pdays_num = pd.to_numeric(X["pdays"], errors="coerce").fillna(-1)
        contacted = ~pdays_num.isin([-1, 999])
        if "previously_contacted" not in X.columns:
            X["previously_contacted"] = contacted.astype(int)
        X["pdays"] = pd.to_numeric(X["pdays"], errors="coerce").where(contacted, 0)
        if "pdays_clean" not in X.columns:
            X["pdays_clean"] = pd.to_numeric(X["pdays"], errors="coerce").where(contacted, 0)
        if "recent_contact_30d" not in X.columns:
            X["recent_contact_30d"] = (contacted & (X["pdays_clean"] <= 30)).astype(int)

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
        if "has_any_loan" not in X.columns:
            X["has_any_loan"] = (has_housing | has_personal).astype(int)
        if "has_both_loans" not in X.columns:
            X["has_both_loans"] = (has_housing & has_personal).astype(int)
        if "debt_free" not in X.columns:
            X["debt_free"] = ((~has_housing) & (~has_personal)).astype(int)

    if "balance" in X.columns:
        bal = pd.to_numeric(X["balance"], errors="coerce").fillna(0.0)
        if "balance_is_negative" not in X.columns:
            X["balance_is_negative"] = (bal < 0).astype(int)
        if "signed_log_balance" not in X.columns:
            X["signed_log_balance"] = np.sign(bal) * np.log1p(np.abs(bal))
        if "balance_to_age_ratio" not in X.columns and "age" in X.columns:
            age = pd.to_numeric(X["age"], errors="coerce").fillna(40.0)
            X["balance_to_age_ratio"] = (bal / (age.clip(lower=18.0) + 1.0)).round(4)

    if "age" in X.columns and "age_group" not in X.columns:
        age = pd.to_numeric(X["age"], errors="coerce").fillna(40.0)
        X["age_group"] = pd.cut(
            age,
            bins=[0, 29, 59, 120],
            labels=["young", "working_age", "senior"],
            include_lowest=True,
        ).astype(str)

    if "job" in X.columns:
        job_clean = X["job"].astype(str).str.strip().str.lower()
        if "is_student_or_retired" not in X.columns:
            X["is_student_or_retired"] = job_clean.isin(["student", "retired"]).astype(int)
        if "is_blue_collar_or_admin" not in X.columns:
            X["is_blue_collar_or_admin"] = job_clean.isin(["blue-collar", "admin."]).astype(int)

    if "education" in X.columns and "education_ordinal" not in X.columns:
        edu_clean = X["education"].astype(str).str.strip().str.lower()
        X["education_ordinal"] = edu_clean.map(EDUCATION_RANK).fillna(0).astype(int)

    if "campaign" in X.columns:
        camp = pd.to_numeric(X["campaign"], errors="coerce").fillna(1.0).clip(lower=1.0)
        if "log_campaign" not in X.columns:
            X["log_campaign"] = np.log1p(camp)
        if "heavy_campaign_calls" not in X.columns:
            X["heavy_campaign_calls"] = (camp >= 4).astype(int)
        if "previous" in X.columns:
            prev = pd.to_numeric(X["previous"], errors="coerce").fillna(0.0).clip(lower=0.0)
            if "total_contacts" not in X.columns:
                X["total_contacts"] = camp + prev
            if "campaign_to_total_ratio" not in X.columns:
                X["campaign_to_total_ratio"] = (camp / (camp + prev + 1.0)).round(4)

    if "month" in X.columns and ("month_sin" not in X.columns or "month_cos" not in X.columns):
        month_num = (
            X["month"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str[:3]
            .map(MONTH_MAP)
            .fillna(5)
        )
        if "month_sin" not in X.columns:
            X["month_sin"] = np.sin(2.0 * np.pi * month_num / 12.0).round(6)
        if "month_cos" not in X.columns:
            X["month_cos"] = np.cos(2.0 * np.pi * month_num / 12.0).round(6)

    if "day" in X.columns and ("day_sin" not in X.columns or "day_cos" not in X.columns):
        day_num = pd.to_numeric(X["day"], errors="coerce").fillna(15).clip(1, 31)
        if "day_sin" not in X.columns:
            X["day_sin"] = np.sin(2.0 * np.pi * day_num / 31.0).round(6)
        if "day_cos" not in X.columns:
            X["day_cos"] = np.cos(2.0 * np.pi * day_num / 31.0).round(6)

    if "poutcome" in X.columns:
        pout = X["poutcome"].astype(str).str.strip().str.lower()
        if "prev_outcome_success" not in X.columns:
            X["prev_outcome_success"] = pout.eq("success").astype(int)
        if "success_per_campaign_call" not in X.columns and "campaign" in X.columns:
            camp = pd.to_numeric(X["campaign"], errors="coerce").fillna(1.0)
            X["success_per_campaign_call"] = (
                X["prev_outcome_success"] / (camp.clip(lower=1.0))
            ).round(4)

    if "balance" in X.columns and "loan_free_positive_balance" not in X.columns:
        bal = pd.to_numeric(X["balance"], errors="coerce").fillna(0.0)
        any_loan = (
            X["has_any_loan"]
            if "has_any_loan" in X.columns
            else (has_housing | has_personal).astype(int)
        )
        X["loan_free_positive_balance"] = ((any_loan == 0) & (bal > 0)).astype(int)

    return X


def model_columns(model) -> list:
    cols = getattr(model, "feature_names_in_", None)
    return list(cols) if cols is not None else RAW_INPUTS + DERIVED


def input_columns(model) -> list:
    return [c for c in model_columns(model) if c not in DERIVED]


def prepare(df: pd.DataFrame, model) -> pd.DataFrame:
    return add_features(df)[model_columns(model)]


def score_positive(model, X: pd.DataFrame):
    classes = list(model.classes_)
    positive = 1 if 1 in classes else POSITIVE_LABEL
    pos_idx = classes.index(positive)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, pos_idx], True
    decision = model.decision_function(X)
    if pos_idx == 0:
        decision = -decision
    return 1.0 / (1.0 + np.exp(-decision)), False


def verdict_for(score: float, threshold: float):
    if score >= threshold:
        return "Likely to subscribe", "#10b981"
    if score >= max(threshold - 0.15, 0.0):
        return "Borderline", "#f59e0b"
    return "Unlikely to subscribe", "#f43f5e"


def advice_for(label: str) -> str:
    if label == "Likely to subscribe":
        return "Prioritise this customer: schedule a call soon and offer the term deposit."
    if label == "Borderline":
        return "Worth a follow-up if capacity allows; a tailored offer could tip the decision."
    return "Low priority for this campaign; consider other products or a later contact."


def ring_html(score: float, color: str) -> str:
    pct = int(round(score * 100))
    return (
        f'<div class="ring-wrap"><div class="ring" style="--p:{pct}; --c:{color}"></div>'
        f'<div class="ring-text"><span class="pct">{pct}%</span>'
        f'<span class="lbl">subscribe score</span></div></div>'
    )


def normalize_binary_target(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map(lambda v: "yes" if v in {"1", "yes", "true"} else "no")


def get_reference_df() -> pd.DataFrame | None:
    if Path(REFERENCE_DATA_PATH).exists():
        return pd.read_csv(REFERENCE_DATA_PATH)
    if (BASE_DIR / "data" / "processed" / "X_train.csv").exists():
        return pd.read_csv(BASE_DIR / "data" / "processed" / "X_train.csv")
    if (BASE_DIR / "data" / "raw" / "bank.csv").exists():
        raw_df = pd.read_csv(BASE_DIR / "data" / "raw" / "bank.csv").drop_duplicates()
        return add_features(raw_df.drop(columns=[TARGET, "duration"], errors="ignore"))
    rng = np.random.RandomState(42)
    n_rows = 250
    return add_features(pd.DataFrame({
        "age": rng.normal(41.0, 10.0, n_rows).clip(18, 95).round().astype(int),
        "job": rng.choice(OPTIONS["job"][:6], n_rows),
        "marital": rng.choice(OPTIONS["marital"], n_rows, p=[0.6, 0.28, 0.12]),
        "education": rng.choice(OPTIONS["education"][:3], n_rows, p=[0.15, 0.50, 0.35]),
        "default": rng.choice(["no", "yes"], n_rows, p=[0.98, 0.02]),
        "balance": rng.normal(1400.0, 900.0, n_rows).round(2),
        "housing": rng.choice(["yes", "no"], n_rows, p=[0.55, 0.45]),
        "loan": rng.choice(["no", "yes"], n_rows, p=[0.84, 0.16]),
        "contact": rng.choice(OPTIONS["contact"], n_rows, p=[0.65, 0.07, 0.28]),
        "day": rng.randint(1, 29, n_rows),
        "month": rng.choice(OPTIONS["month"], n_rows),
        "campaign": rng.poisson(2.0, n_rows) + 1,
        "pdays": np.zeros(n_rows, dtype=int),
        "previous": np.zeros(n_rows, dtype=int),
        "poutcome": rng.choice(OPTIONS["poutcome"], n_rows, p=[0.80, 0.11, 0.04, 0.05]),
    }))


@st.cache_resource(show_spinner=False)
def load_champion_model(path_str: str, mtime: float):
    return joblib.load(path_str)


@st.cache_resource(show_spinner=False)
def load_model_from_bytes(data: bytes):
    return joblib.load(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def load_reports():
    def _read(path: Path):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    return _read(SELECTION_PATH), _read(METRICS_PATH), _read(DETAILS_PATH)


def init_state():
    for col, value in DEFAULT_VALUES.items():
        st.session_state.setdefault(f"in_{col}", value)


def apply_preset():
    preset = PRESETS.get(st.session_state.get("preset"))
    if preset:
        for col, value in preset.items():
            st.session_state[f"in_{col}"] = value


# --------------------------------------------------------------------------
# Page Setup & Sidebar
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing MLOps Portal",
    page_icon="🏦",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)
init_state()

selection, metrics, details = load_reports()

with st.sidebar:
    st.title("🏦 MLOps Studio")
    st.caption("Term Deposit Propensity & Drift Suite")

    if selection and "champion" in selection:
        champ = selection["champion"]
        crit = champ.get("criteria", {})
        primary_metric = selection.get("primary_metric", "f1").upper()

        st.subheader("🏆 Active Champion")
        st.success(f"**{champ.get('display_name', champ.get('model_name'))}**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(f"Val {primary_metric}", f"{crit.get('validation_score_pct', 0.0):.1f}%")
            st.metric("Latency", f"{crit.get('prediction_time_ms', 0.0):.2f} ms")
        with col_b:
            if metrics:
                st.metric(f"Test {primary_metric}", f"{metrics.get('f1', 0.0) * 100:.1f}%")
            st.metric("Size", f"{crit.get('model_size_mb', 0.0):.3f} MB")

        with st.expander("Selection Rationale"):
            st.write(champ.get("explanation", "Selected via multi-criteria rules."))
    else:
        st.warning("Model metadata not found. Run `dvc repro` to build.")

    st.divider()
    st.subheader("📡 Live Monitoring")
    log_df_sidebar = load_inference_logs(INFERENCE_LOG_PATH)
    st.metric("Logged Inferences", len(log_df_sidebar))

    st.divider()
    st.subheader("⚙️ Inference Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Customers scoring at or above this threshold are flagged as likely subscribers.",
    )
    uploaded_model = st.file_uploader(
        "Override model (.pkl)",
        type=["pkl", "joblib"],
        help="Optional: upload a fitted scikit-learn pipeline trained on the bank dataset.",
    )
    st.caption("DVC • MLflow • Optuna • KS & Chi-Square Drift")

model = None
model_source = ""
if uploaded_model is not None:
    model = load_model_from_bytes(uploaded_model.getvalue())
    model_source = f"uploaded ({uploaded_model.name})"
elif MODEL_PATH.exists():
    model = load_champion_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    model_source = "models/model.pkl"

st.markdown(
    '<div class="hero">'
    '<div class="hero-tag">Production ML & Drift Intelligence</div>'
    '<h1>🏦 Bank Marketing MLOps Portal</h1>'
    "<p>Score term deposit propensity with key customer signals, inspect champion lineage, and track live data drift.</p>"
    "</div>",
    unsafe_allow_html=True,
)

tab_predict, tab_batch, tab_metrics, tab_monitoring = st.tabs([
    "🔮 Live Inference",
    "📂 Batch Scoring",
    "📊 Model Lineage & Metrics",
    "🛡️ Data Drift & Monitoring",
])

# ------------------ TAB 1: LIVE INFERENCE ------------------
with tab_predict:
    st.header("Term Deposit Subscription Prediction")
    st.caption("Showing key customer, financial, and campaign features. Secondary fields use safe defaults automatically.")

    if model is None:
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. "
            "Run `dvc repro` to train and select the champion model, or upload one in the sidebar."
        )
        st.stop()

    all_input_cols = input_columns(model)
    unsupported = [c for c in all_input_cols if c not in ALLOWED_COLUMNS]
    if unsupported:
        st.error(f"Model expects unsupported input columns: {unsupported}")
        st.stop()

    active_form_cols = [c for c in CORE_FORM_INPUTS if c in all_input_cols]

    st.selectbox("Quick start profile", list(PRESETS.keys()), key="preset", on_change=apply_preset)

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="group-card group-profile">👤 Customer Profile</div>', unsafe_allow_html=True)
            if "age" in active_form_cols:
                st.number_input("Age", min_value=18, max_value=100, step=1, key="in_age")
            if "job" in active_form_cols:
                st.selectbox("Job", OPTIONS["job"], key="in_job")
            if "marital" in active_form_cols:
                st.selectbox("Marital status", OPTIONS["marital"], key="in_marital")
            if "education" in active_form_cols:
                st.selectbox("Education", OPTIONS["education"], key="in_education")

        with c2:
            st.markdown('<div class="group-card group-finance">💰 Financial Standing</div>', unsafe_allow_html=True)
            if "balance" in active_form_cols:
                st.number_input("Average yearly balance", min_value=-10000, max_value=200000, step=100, key="in_balance")
            if "housing" in active_form_cols:
                st.selectbox("Housing loan?", OPTIONS["housing"], key="in_housing")
            if "loan" in active_form_cols:
                st.selectbox("Personal loan?", OPTIONS["loan"], key="in_loan")

        with c3:
            st.markdown('<div class="group-card group-campaign">📞 Campaign Signals</div>', unsafe_allow_html=True)
            if "contact" in active_form_cols:
                st.selectbox("Contact channel", OPTIONS["contact"], key="in_contact")
            if "month" in active_form_cols:
                st.selectbox("Last contact month", OPTIONS["month"], key="in_month")
            if "campaign" in active_form_cols:
                st.number_input("Contacts in this campaign", min_value=1, max_value=60, step=1, key="in_campaign")
            if "poutcome" in active_form_cols:
                st.selectbox("Previous campaign outcome", OPTIONS["poutcome"], key="in_poutcome")

        submitted = st.form_submit_button("🔮 Predict & Log Inference", type="primary")

    if submitted:
        row = {
            col: st.session_state.get(f"in_{col}", DEFAULT_VALUES[col])
            for col in all_input_cols
        }
        X = prepare(pd.DataFrame([row]), model)
        scores, is_prob = score_positive(model, X)
        score = float(scores[0])
        pred_label = "yes" if score >= threshold else "no"
        label, color = verdict_for(score, threshold)

        proba_dict = {"No": round(1.0 - score, 4), "Yes": round(score, 4)}
        try:
            log_inference(X, pred_label, proba_dict, log_path=INFERENCE_LOG_PATH)
        except Exception as exc:
            st.warning(f"Prediction succeeded, but logging failed: {exc}")

        st.markdown("---")
        left, mid, right = st.columns([1, 1.3, 1.1])
        with left:
            st.markdown(ring_html(score, color), unsafe_allow_html=True)
        with mid:
            st.markdown(
                f'<span class="badge" style="background:{color}">{label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"#### {advice_for(label)}")
            st.markdown(
                f'<p class="small-note">Source: <code>{model_source}</code> | Threshold: {threshold:.0%}. '
                f'{"" if is_prob else "Score is a sigmoid-scaled decision value. "}'
                "Request logged for drift monitoring.</p>",
                unsafe_allow_html=True,
            )
        with right:
            st.write("##### Class Likelihoods")
            proba_df = pd.DataFrame(
                list(proba_dict.items()), columns=["Outcome", "Probability"]
            ).set_index("Outcome")
            st.bar_chart(proba_df)

        with st.expander("Inspect Engineered Feature Payload"):
            st.dataframe(X.T.rename(columns={0: "value"}).astype(str))

# ------------------ TAB 2: BATCH SCORING ------------------
with tab_batch:
    st.header("Batch Campaign Scoring")
    if model is None:
        st.warning("Load or train a model first.")
    else:
        columns = input_columns(model)
        st.markdown("Upload a customer CSV containing these required columns:")
        st.code(", ".join(columns), language="text")

        template = pd.DataFrame([{c: DEFAULT_VALUES[c] for c in columns}])
        st.download_button(
            "⬇️ Download CSV template",
            template.to_csv(index=False).encode("utf-8"),
            file_name="bank_template.csv",
            mime="text/csv",
        )

        upload = st.file_uploader("Upload customers (CSV)", type=["csv"], key="batch_upload")
        if upload is not None:
            df = pd.read_csv(upload)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                X = prepare(df[columns], model)
                scores, _ = score_positive(model, X)
                out = df.copy()
                out["likelihood_score"] = np.round(scores, 4)
                out["prediction"] = np.where(scores >= threshold, POSITIVE_LABEL, "no")

                for i in range(min(len(X), 500)):
                    s_i = float(scores[i])
                    log_inference(
                        X.iloc[[i]],
                        out["prediction"].iloc[i],
                        {"No": round(1.0 - s_i, 4), "Yes": round(s_i, 4)},
                        log_path=INFERENCE_LOG_PATH,
                    )

                n_yes = int((out["prediction"] == POSITIVE_LABEL).sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Customers scored", f"{len(out):,}")
                m2.metric("Predicted subscribers", f"{n_yes:,}")
                m3.metric("Predicted conversion rate", f"{n_yes / len(out):.1%}")

                if TARGET in out.columns:
                    actual = normalize_binary_target(out[TARGET])
                    m4, m5 = st.columns(2)
                    m4.metric("Accuracy vs actual", f"{accuracy_score(actual, out['prediction']):.3f}")
                    m5.metric(
                        "F1 (yes) vs actual",
                        f"{f1_score(actual, out['prediction'], pos_label=POSITIVE_LABEL):.3f}",
                    )

                out = out.sort_values("likelihood_score", ascending=False).reset_index(drop=True)
                st.dataframe(
                    out,
                    column_config={
                        "likelihood_score": st.column_config.ProgressColumn(
                            "Likelihood score", min_value=0.0, max_value=1.0, format="%.2f"
                        )
                    },
                )
                st.download_button(
                    "⬇️ Download predictions",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="bank_predictions.csv",
                    mime="text/csv",
                )

# ------------------ TAB 3: MODEL LINEAGE & METRICS ------------------
with tab_metrics:
    st.header("Model Lineage & Multi-Criteria Benchmark")
    st.write("All 7 candidate models were tuned with Optuna and evaluated across 6 criteria to select the champion:")

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test F1 (yes)", f"{metrics.get('f1', 0.0):.3f}")
        c2.metric("Test ROC-AUC", f"{metrics.get('roc_auc', 0.0):.3f}")
        c3.metric("Test Accuracy", f"{metrics.get('accuracy', 0.0):.3f}")
        c4.metric("Test Precision", f"{metrics.get('precision', 0.0):.3f}")
        c5.metric("Test Recall", f"{metrics.get('recall', 0.0):.3f}")

    if COMPARISON_CSV_PATH.exists():
        st.subheader("Candidate Model Comparison (CV & Runtime Criteria)")
        comp_df = pd.read_csv(COMPARISON_CSV_PATH)
        st.dataframe(comp_df, use_container_width=True)
    elif selection:
        st.json(selection)

    if details:
        all_models = pd.DataFrame(details["all_models_test_metrics"]).T
        left, right = st.columns(2)
        with left:
            st.subheader("Held-Out Test F1 by Model")
            st.bar_chart(all_models[["f1"]])
        with right:
            st.subheader("Champion Confusion Matrix (Test Set)")
            cm = details["champion_confusion_matrix"]
            cm_df = pd.DataFrame(
                [
                    [cm["true_negative"], cm["false_positive"]],
                    [cm["false_negative"], cm["true_positive"]],
                ],
                index=["Actual: no", "Actual: yes"],
                columns=["Predicted: no", "Predicted: yes"],
            )
            st.dataframe(cm_df)

        st.subheader("All Models: Held-Out Test Metrics")
        st.dataframe(all_models.round(4), use_container_width=True)
    elif not COMPARISON_CSV_PATH.exists() and not metrics:
        st.info("Run `dvc repro` to generate model lineage and evaluation reports.")

# ------------------ TAB 4: DATA DRIFT & MONITORING ------------------
with tab_monitoring:
    st.header("Production Data Drift & Quality Monitoring")
    st.write(
        "Monitoring production inputs against the reference training dataset using "
        "**Kolmogorov-Smirnov (numeric)** and **Chi-Square (categorical)** two-sample tests."
    )

    log_df = load_inference_logs(INFERENCE_LOG_PATH)
    ref_df = get_reference_df()

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1.2, 1.9, 1.9, 1.2])
    with col_btn1:
        run_check = st.button(
            "Run Drift Check",
            type="primary",
            disabled=log_df.empty,
            help="Requires at least 1 logged inference.",
        )
    with col_btn2:
        seed_clean = st.button(
            "🧪 Log 100 Clean Requests",
            disabled=(model is None or ref_df is None),
            help="Sample 100 baseline customer profiles from training distribution.",
        )
    with col_btn3:
        seed_drift = st.button(
            "🚨 Simulate 100 Drifted",
            disabled=(model is None or ref_df is None),
            help="Synthesize 100 shifted customer profiles to test drift detection.",
        )
    with col_btn4:
        clear_logs = st.button(
            "🗑️ Clear Logs",
            disabled=log_df.empty,
            help="Delete reports/inference_logs.csv to reset monitoring.",
        )

    if clear_logs:
        log_path = Path(INFERENCE_LOG_PATH)
        if log_path.exists():
            log_path.unlink()
        st.rerun()

    if (seed_clean or seed_drift) and model is not None and ref_df is not None:
        sample_X = generate_sample_production_data(
            ref_df, simulate_drift=bool(seed_drift), n_samples=100, seed=42
        )
        sample_X = prepare(sample_X, model)
        scores, _ = score_positive(model, sample_X)
        for i in range(len(sample_X)):
            s_i = float(scores[i])
            pred_i = "yes" if s_i >= threshold else "no"
            log_inference(
                sample_X.iloc[[i]],
                pred_i,
                {"No": round(1.0 - s_i, 4), "Yes": round(s_i, 4)},
                log_path=INFERENCE_LOG_PATH,
            )
        st.rerun()

    if not log_df.empty:
        st.write(f"**Total production records logged:** `{len(log_df)}`")

        if run_check or len(log_df) >= 5:
            if ref_df is not None:
                try:
                    report = analyze_drift(ref_df, log_df)

                    status_color = "red" if report["dataset_drift_detected"] else "green"
                    status_msg = "⚠️ DRIFT DETECTED" if report["dataset_drift_detected"] else "✅ HEALTHY DISTRIBUTION"

                    st.markdown(f"#### Status: :{status_color}[{status_msg}]")
                    st.write(
                        f"Drifted features: **{report['drifted_features']} / {report['total_features']}** "
                        f"({report['drift_share_pct']}%)"
                    )

                    drift_rows = []
                    for feat, d in report["features"].items():
                        stat_val = d.get("ks_statistic", d.get("chi2_statistic", d.get("statistic", 0.0)))
                        ref_mean = d.get("reference_mean")
                        cur_mean = d.get("current_mean")
                        shift = d.get("mean_shift")
                        drift_rows.append({
                            "Feature": feat,
                            "Test": d.get("test_name", "KS / Chi2"),
                            "Status": "DRIFT ⚠️" if d["drift_detected"] else "HEALTHY ✅",
                            "p-value": round(float(d["p_value"]), 4),
                            "Statistic": round(float(stat_val), 4) if stat_val is not None else "N/A",
                            "Reference Mean": round(float(ref_mean), 2) if ref_mean is not None else "Categorical",
                            "Production Mean": round(float(cur_mean), 2) if cur_mean is not None else "Categorical",
                            "Mean Shift": round(float(shift), 2) if shift is not None else "N/A",
                        })
                    st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)
                except Exception as exc:
                    st.error(f"Drift analysis failed: {exc}")
            else:
                st.info("Reference dataset not found (`data/processed/X_train.csv` or `data/raw/bank.csv`).")

        with st.expander("View Recent Production Logs (Last 10 Requests)"):
            st.dataframe(log_df.tail(10), use_container_width=True)
    else:
        st.info(
            "No production inferences logged yet. Submit predictions in **Live Inference**, "
            "upload a CSV in **Batch Scoring**, or click **🧪 Log 100 Clean Requests** above."
        )