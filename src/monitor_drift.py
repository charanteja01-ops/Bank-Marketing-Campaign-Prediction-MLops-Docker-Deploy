"""Statistical Data Drift Monitoring for Bank Marketing Campaign Prediction.

Supports:
- Inference payload & prediction logging (`log_inference`, `load_inference_logs`).
- Two-sample Kolmogorov-Smirnov (numeric) and Chi-Square (categorical) drift tests (`analyze_drift`).
- Structured Markdown alert report generation (`generate_alert_markdown`) and webhook dispatch (`send_drift_alert`).
- CLI execution for scheduled GitHub Actions continuous training workflows.
"""
import argparse
import json
import math
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None

ROOT_DIR = Path(__file__).resolve().parents[1]
INFERENCE_LOG_PATH = ROOT_DIR / "reports" / "inference_logs.csv"
REFERENCE_DATA_PATH = ROOT_DIR / "data" / "processed" / "X_train.csv"
DRIFT_REPORT_PATH = ROOT_DIR / "reports" / "drift_report.json"
DRIFT_ALERT_PATH = ROOT_DIR / "reports" / "drift_alert.md"

# 16 engineered features matching src/feature_engineering.py, X_train.csv, and models/model.pkl
FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "previously_contacted",
]

# Continuous / count numeric features evaluated with two-sample KS-test.
# Binary indicator 'previously_contacted' (0/1) is evaluated with Chi-Square.
NUMERIC_FEATURES = {
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
}


def _ks_2samp(x, y):
    if _scipy_stats is not None:
        stat, p_val = _scipy_stats.ks_2samp(x, y)
        return float(stat), float(p_val)
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
    if _scipy_stats is not None:
        stat, p_val, _, _ = _scipy_stats.chi2_contingency(contingency)
        return float(stat), float(p_val)
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


def log_inference(input_df, prediction, proba_dict=None, log_path=INFERENCE_LOG_PATH):
    """Append production input features, predicted class, and probabilities to CSV."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in FEATURE_COLUMNS if c in input_df.columns]
    record = input_df[cols].copy() if cols else input_df.copy()
    record["predicted_class"] = prediction
    if proba_dict:
        for label, prob in proba_dict.items():
            record[f"prob_{label}"] = float(prob)
    record["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()

    write_header = not path.exists()
    record.to_csv(path, mode="a", header=write_header, index=False)


def load_inference_logs(log_path=INFERENCE_LOG_PATH):
    """Load logged production inferences if available."""
    path = Path(log_path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def analyze_drift(ref_df, curr_df, p_value_threshold=0.05, drift_share_threshold=0.50):
    """Compare production data against reference training data.

    Uses the two-sample Kolmogorov-Smirnov (KS) test for continuous numeric features
    and the Chi-Square contingency test for categorical/binary features.
    """
    available_features = [
        c for c in FEATURE_COLUMNS if c in ref_df.columns and c in curr_df.columns
    ]
    if not available_features:
        raise ValueError("No overlapping feature columns found between reference and current data.")

    feature_results = {}
    drifted_count = 0

    for col in available_features:
        ref_series = ref_df[col].dropna()
        curr_series = curr_df[col].dropna()

        if len(ref_series) == 0 or len(curr_series) == 0:
            feature_results[col] = {
                "test_name": "Skipped",
                "drift_detected": False,
                "p_value": 1.0,
                "statistic": 0.0,
                "reference_mean": None,
                "current_mean": None,
                "mean_shift": None,
                "reference_summary": "N/A",
                "current_summary": "N/A",
                "shift_summary": "N/A",
            }
            continue

        is_numeric = col in NUMERIC_FEATURES

        if is_numeric:
            ref_vals = pd.to_numeric(ref_series, errors="coerce").dropna()
            curr_vals = pd.to_numeric(curr_series, errors="coerce").dropna()
            ks_stat, p_val = _ks_2samp(ref_vals, curr_vals)
            drifted = bool(p_val < p_value_threshold)
            ref_mean = float(ref_vals.mean())
            ref_std = float(ref_vals.std(ddof=0))
            curr_mean = float(curr_vals.mean())
            curr_std = float(curr_vals.std(ddof=0))
            shift = float(curr_mean - ref_mean)

            feature_results[col] = {
                "test_name": "KS-test",
                "drift_detected": drifted,
                "p_value": float(p_val),
                "ks_statistic": float(ks_stat),
                "statistic": float(ks_stat),
                "reference_mean": ref_mean,
                "current_mean": curr_mean,
                "mean_shift": shift,
                "reference_summary": f"`{ref_mean:.2f} (±{ref_std:.2f})`",
                "current_summary": f"`{curr_mean:.2f} (±{curr_std:.2f})`",
                "shift_summary": f"`{shift:+.2f}`",
            }
        else:
            ref_counts = ref_series.astype(str).value_counts()
            curr_counts = curr_series.astype(str).value_counts()
            categories = sorted(set(ref_counts.index) | set(curr_counts.index))

            if len(categories) < 2:
                chi2_stat, p_val = 0.0, 1.0
            else:
                contingency = np.array([
                    [ref_counts.get(cat, 0) for cat in categories],
                    [curr_counts.get(cat, 0) for cat in categories],
                ])
                chi2_stat, p_val = _chi2_contingency(contingency)

            ref_top = ref_counts.index[0]
            ref_top_pct = (ref_counts.iloc[0] / len(ref_series)) * 100.0
            curr_top = curr_counts.index[0]
            curr_top_pct = (curr_counts.iloc[0] / len(curr_series)) * 100.0
            ref_share_of_curr_top = (ref_counts.get(curr_top, 0) / len(ref_series)) * 100.0
            pp_shift = curr_top_pct - ref_share_of_curr_top

            drifted = bool(p_val < p_value_threshold)
            feature_results[col] = {
                "test_name": "Chi-square",
                "drift_detected": drifted,
                "p_value": float(p_val),
                "chi2_statistic": float(chi2_stat),
                "statistic": float(chi2_stat),
                "reference_mean": None,
                "current_mean": None,
                "mean_shift": None,
                "reference_summary": f"`top: {ref_top} ({ref_top_pct:.0f}%)`",
                "current_summary": f"`top: {curr_top} ({curr_top_pct:.0f}%)`",
                "shift_summary": f"`{curr_top} {pp_shift:+.1f} pp`",
            }

        if drifted:
            drifted_count += 1

    total = len(available_features)
    drift_share = round((drifted_count / total) * 100.0, 2) if total > 0 else 0.0
    dataset_drift = bool((drifted_count / total) >= drift_share_threshold) if total > 0 else False

    return {
        "evaluation_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_drift_detected": dataset_drift,
        "drifted_features": drifted_count,
        "total_features": total,
        "drift_share_pct": drift_share,
        "p_value_threshold": p_value_threshold,
        "drift_share_threshold_pct": round(drift_share_threshold * 100.0, 2),
        "reference_samples": int(len(ref_df)),
        "production_samples": int(len(curr_df)),
        "features": feature_results,
    }


def generate_alert_markdown(report, output_path=DRIFT_ALERT_PATH):
    """Write a structured Markdown drift summary report for GitHub Issues & Step Summary."""
    drifted = report["dataset_drift_detected"]
    title = (
        "# 🚨 **HIGH PRIORITY: PRODUCTION DATA DRIFT DETECTED**"
        if drifted
        else "# ✅ **PRODUCTION DATA DISTRIBUTION HEALTHY**"
    )
    ts = report.get("evaluation_timestamp", pd.Timestamp.now(tz="UTC").isoformat())
    threshold_pct = report.get("drift_share_threshold_pct", 50.0)
    ref_n = report.get("reference_samples", "N/A")
    prod_n = report.get("production_samples", "N/A")

    lines = [
        title,
        "",
        "### 📊 Monitoring Summary",
        f"- **Evaluation Timestamp:** `{ts}`",
        f"- **Drifted Features:** **{report['drifted_features']} of {report['total_features']}** ({report['drift_share_pct']}%)",
        f"- **Dataset Drift Threshold:** `{threshold_pct}%`",
        f"- **Reference Training Samples:** `{ref_n}`",
        f"- **Production Inference Samples:** `{prod_n}`",
        "",
        "### 🔬 Statistical Tests (KS for numeric, Chi-square for categorical)",
        "| Feature | Test | Status | p-value | Statistic | Reference | Production | Shift |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for feat, info in report.get("features", {}).items():
        status = "⚠️ **DRIFT**" if info["drift_detected"] else "✅ Healthy"
        lines.append(
            f"| `{feat}` | {info['test_name']} | {status} | `{info['p_value']:.4f}` | "
            f"`{info['statistic']:.4f}` | {info['reference_summary']} | "
            f"{info['current_summary']} | {info['shift_summary']} |"
        )

    lines.extend(["", "---", "### ⚙️ Action & Continuous Training Status"])
    if drifted:
        lines.append(
            "⚠️ **Continuous Training Required:** Production feature distributions have deviated "
            "significantly from the baseline training distribution. Triggering automatic pipeline "
            "reproduction (`dvc repro`) and quality gate evaluation."
        )
    else:
        lines.append(
            "✅ **No Action Required:** Production feature distributions remain consistent with "
            "the baseline training distribution."
        )

    md_text = "\n".join(lines) + "\n"
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md_text, encoding="utf-8")
    return md_text


def send_drift_alert(report, webhook_url=None):
    """Post a JSON drift notification to a webhook (Slack/Teams/Discord) if configured."""
    if not webhook_url or not report.get("dataset_drift_detected", False):
        return False

    payload = json.dumps({
        "text": (
            f"🚨 Bank Marketing Data Drift Detected: "
            f"{report['drifted_features']}/{report['total_features']} features drifted "
            f"({report['drift_share_pct']}%)."
        ),
        "report": {
            "drifted_features": report["drifted_features"],
            "total_features": report["total_features"],
            "drift_share_pct": report["drift_share_pct"],
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def generate_sample_production_data(ref_df, simulate_drift=False, n_samples=250, seed=42):
    """Sample reference data or synthesize shifted production traffic for scheduled CI checks."""
    rng = np.random.RandomState(seed)
    sample = ref_df.sample(n=min(n_samples, len(ref_df)), replace=True, random_state=seed).copy()

    if simulate_drift:
        if "age" in sample.columns:
            sample["age"] = rng.normal(70, 5, len(sample)).clip(58, 95).round().astype(int)
        if "balance" in sample.columns:
            sample["balance"] = rng.normal(18000, 2500, len(sample)).round(2)
        if "campaign" in sample.columns:
            sample["campaign"] = rng.randint(10, 25, len(sample))
        if "job" in sample.columns:
            sample["job"] = rng.choice(["retired", "student"], len(sample), p=[0.8, 0.2])
        if "month" in sample.columns:
            sample["month"] = rng.choice(["dec", "mar", "oct"], len(sample))
        if "poutcome" in sample.columns:
            sample["poutcome"] = "success"

    return sample.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Run statistical data drift analysis.")
    parser.add_argument("--generate-sample-data", action="store_true")
    parser.add_argument("--simulate-drift", action="store_true")
    parser.add_argument("--reference-path", default=str(REFERENCE_DATA_PATH))
    parser.add_argument("--current-path", default=str(INFERENCE_LOG_PATH))
    parser.add_argument("--report-path", default=str(DRIFT_REPORT_PATH))
    parser.add_argument("--alert-path", default=str(DRIFT_ALERT_PATH))
    parser.add_argument("--webhook-url", default=os.environ.get("ALERT_WEBHOOK_URL"))
    args = parser.parse_args()

    ref_path = Path(args.reference_path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference dataset not found: {ref_path}")
    ref_df = pd.read_csv(ref_path)

    if args.generate_sample_data:
        curr_df = generate_sample_production_data(ref_df, simulate_drift=args.simulate_drift)
    else:
        curr_df = load_inference_logs(args.current_path)
        if curr_df.empty:
            print("No production inference logs found; falling back to clean sample.")
            curr_df = generate_sample_production_data(ref_df, simulate_drift=False)

    report = analyze_drift(ref_df, curr_df)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    generate_alert_markdown(report, output_path=args.alert_path)
    send_drift_alert(report, webhook_url=args.webhook_url)

    print(
        f"Drift check complete: detected={report['dataset_drift_detected']} "
        f"({report['drifted_features']}/{report['total_features']} features, "
        f"{report['drift_share_pct']}%)"
    )


if __name__ == "__main__":
    main()
