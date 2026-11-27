from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering import add_features

MODEL_PATH = Path("models/model.pkl")

RAW_INPUTS = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "campaign", "pdays", "previous", "poutcome",
]
VALID_CLASSES = {0, 1}


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return add_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Champion model not found at {MODEL_PATH}. Run `dvc repro` first.")
    return joblib.load(MODEL_PATH)


def test_model_pipeline_structure(model):
    """Ensure the champion is a Pipeline with Preprocessor and Model stages."""
    assert hasattr(model, "steps"), "Model should be a scikit-learn Pipeline"
    step_names = [name for name, _ in model.steps]
    assert "Preprocessor" in step_names, "Pipeline missing 'Preprocessor' stage"
    assert "Model" in step_names, "Pipeline missing 'Model' stage"

    if hasattr(model, "feature_names_in_"):
        sample = _make_df([{
            "age": 41, "job": "management", "marital": "married",
            "education": "secondary", "default": "no", "balance": 1500,
            "housing": "yes", "loan": "no", "contact": "cellular",
            "day": 15, "month": "may", "campaign": 2,
            "pdays": -1, "previous": 0, "poutcome": "unknown",
        }])
        assert list(model.feature_names_in_) == list(sample.columns)
    assert set(model.classes_) == VALID_CLASSES


def test_single_row_inference(model):
    """Test inference on a typical customer profile."""
    sample = _make_df([{
        "age": 34, "job": "management", "marital": "single",
        "education": "tertiary", "default": "no", "balance": 3500,
        "housing": "no", "loan": "no", "contact": "cellular",
        "day": 15, "month": "mar", "campaign": 1,
        "pdays": 90, "previous": 2, "poutcome": "success",
    }])
    pred = model.predict(sample)
    assert len(pred) == 1
    assert int(pred[0]) in VALID_CLASSES


def test_batch_inference_and_probabilities(model):
    """Test batch inference and verify probability distributions sum to ~1.0."""
    batch = _make_df([
        {
            "age": 34, "job": "management", "marital": "single",
            "education": "tertiary", "default": "no", "balance": 3500,
            "housing": "no", "loan": "no", "contact": "cellular",
            "day": 15, "month": "mar", "campaign": 1,
            "pdays": 90, "previous": 2, "poutcome": "success",
        },
        {
            "age": 67, "job": "retired", "marital": "married",
            "education": "secondary", "default": "no", "balance": 5200,
            "housing": "no", "loan": "no", "contact": "cellular",
            "day": 10, "month": "oct", "campaign": 1,
            "pdays": -1, "previous": 0, "poutcome": "unknown",
        },
        {
            "age": 41, "job": "blue-collar", "marital": "married",
            "education": "primary", "default": "no", "balance": 250,
            "housing": "yes", "loan": "yes", "contact": "unknown",
            "day": 20, "month": "may", "campaign": 3,
            "pdays": -1, "previous": 0, "poutcome": "unknown",
        },
    ])

    preds = model.predict(batch)
    assert len(preds) == 3
    for p in preds:
        assert int(p) in VALID_CLASSES

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(batch)
        assert probas.shape == (3, 2)
        for row in probas:
            assert np.isclose(np.sum(row), 1.0, atol=1e-5)


def test_prediction_determinism(model):
    """Ensure identical inputs yield identical outputs (deterministic)."""
    sample = _make_df([{
        "age": 41, "job": "management", "marital": "married",
        "education": "secondary", "default": "no", "balance": 1500,
        "housing": "yes", "loan": "no", "contact": "cellular",
        "day": 15, "month": "may", "campaign": 2,
        "pdays": -1, "previous": 0, "poutcome": "unknown",
    }])
    pred_1 = int(model.predict(sample)[0])
    pred_2 = int(model.predict(sample)[0])
    assert pred_1 == pred_2


def test_handles_missing_and_unseen_categories(model):
    """Verify SimpleImputer and OneHotEncoder(handle_unknown='ignore') handle edge cases."""
    sample = _make_df([{
        "age": np.nan, "job": "new_unseen_job", "marital": "married",
        "education": np.nan, "default": "no", "balance": np.nan,
        "housing": "yes", "loan": "no", "contact": "cellular",
        "day": 15, "month": "may", "campaign": 2,
        "pdays": -1, "previous": 0, "poutcome": "unknown",
    }])
    sample["age_group"] = "unseen_age_group"
    pred = model.predict(sample)
    assert len(pred) == 1
    assert int(pred[0]) in VALID_CLASSES