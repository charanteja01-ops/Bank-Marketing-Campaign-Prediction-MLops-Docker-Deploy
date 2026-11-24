from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path("models/model.pkl")

# Exact feature order produced by src/feature_engineering.py (with 'duration' dropped)
FEATURES = [
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
VALID_CLASSES = {0, 1}


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
        assert list(model.feature_names_in_) == FEATURES
    assert set(model.classes_) == VALID_CLASSES


def test_single_row_inference(model):
    """Test inference on a typical customer profile."""
    sample = pd.DataFrame(
        [[34, "management", "single", "tertiary", "no", 3500, "no", "no",
          "cellular", 15, "mar", 1, 90, 2, "success", 1]],
        columns=FEATURES,
    )
    pred = model.predict(sample)
    assert len(pred) == 1
    assert int(pred[0]) in VALID_CLASSES


def test_batch_inference_and_probabilities(model):
    """Test batch inference and verify probability distributions sum to ~1.0."""
    batch = pd.DataFrame([
        [34, "management", "single", "tertiary", "no", 3500, "no", "no",
         "cellular", 15, "mar", 1, 90, 2, "success", 1],
        [67, "retired", "married", "secondary", "no", 5200, "no", "no",
         "cellular", 10, "oct", 1, 0, 0, "unknown", 0],
        [41, "blue-collar", "married", "primary", "no", 250, "yes", "yes",
         "unknown", 20, "may", 3, 0, 0, "unknown", 0],
    ], columns=FEATURES)

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
    sample = pd.DataFrame(
        [[41, "management", "married", "secondary", "no", 1500, "yes", "no",
          "cellular", 15, "may", 2, 0, 0, "unknown", 0]],
        columns=FEATURES,
    )
    pred_1 = int(model.predict(sample)[0])
    pred_2 = int(model.predict(sample)[0])
    assert pred_1 == pred_2


def test_handles_missing_and_unseen_categories(model):
    """Verify SimpleImputer and OneHotEncoder(handle_unknown='ignore') handle edge cases."""
    sample = pd.DataFrame(
        [[np.nan, "new_unseen_job", "married", np.nan, "no", np.nan, "yes", "no",
          "cellular", 15, "may", 2, 0, 0, "unknown", 0]],
        columns=FEATURES,
    )
    pred = model.predict(sample)
    assert len(pred) == 1
    assert int(pred[0]) in VALID_CLASSES
