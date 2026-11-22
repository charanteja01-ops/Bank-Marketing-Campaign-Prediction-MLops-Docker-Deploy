"""Final evaluation of the automatically selected champion on the untouched
test set (Bank Marketing Campaign Prediction).

The champion is chosen in model_building.py using ONLY training data
(cross-validation + size/time criteria). This script never changes that choice;
test metrics for the other models are reported for comparison only.
"""
from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    get_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
SELECTION_PATH = Path("reports/model_selection.json")
COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
METRICS_PATH = Path("reports/metrics.json")
DETAILS_PATH = Path("reports/evaluation_details.json")

MODEL_NAMES = [
    "KNN",
    "DecisionTree",
    "LogisticRegression",
    "GaussianNB",
    "RandomForest",
    "GradientBoosting",
]


def positive_scores(model, X):
    """Continuous score for the positive class (needed for ROC-AUC / PR-AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = positive_scores(model, X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
    }


def detailed_report(model, X_test, y_test):
    predictions = model.predict(X_test)
    return classification_report(
        y_test, predictions, output_dict=True, zero_division=0,
        target_names=["no", "yes"],
    )


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    target = config["data"]["target"]
    metric_name = config["metric"]["primary"]
    scorer = get_scorer(metric_name)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[target].astype(int)

    with open(SELECTION_PATH, "r", encoding="utf-8") as file:
        selection = json.load(file)

    champion_info = selection["champion"]
    champion_name = champion_info["model_name"]
    champion_path = Path(selection["champion_model_path"])
    champion = joblib.load(champion_path)
    criteria = champion_info["criteria"]
    policy = selection["selection_policy"]

    # Test metrics for ALL candidates are for comparison/reporting only.
    # They are never used to pick or change the champion.
    all_results = {}
    for model_name in MODEL_NAMES:
        model = joblib.load(Path("models") / f"{model_name}.pkl")
        all_results[model_name] = evaluate_model(model, X_test, y_test)

    champion_metrics = evaluate_model(champion, X_test, y_test)
    test_primary_pct = float(scorer(champion, X_test, y_test)) * 100.0

    tn, fp, fn, tp = confusion_matrix(y_test, champion.predict(X_test)).ravel()

    # Flat, numeric-only file so `dvc metrics show` stays readable.
    metrics = {
        "accuracy": champion_metrics["accuracy"],
        "balanced_accuracy": champion_metrics["balanced_accuracy"],
        "precision": champion_metrics["precision"],
        "recall": champion_metrics["recall"],
        "f1": champion_metrics["f1"],
        "roc_auc": champion_metrics["roc_auc"],
        "pr_auc": champion_metrics["pr_auc"],
        "champion_cv_score": champion_info["best_cv_score"],
        "champion_training_score_pct": criteria["training_score_pct"],
        "champion_validation_score_pct": criteria["validation_score_pct"],
        "champion_score_gap_pct": criteria["score_gap_pct"],
        "champion_model_size_mb": criteria["model_size_mb"],
        "champion_training_time_seconds": criteria["training_time_seconds"],
        "champion_prediction_time_ms": criteria["prediction_time_ms"],
        # Same metric as used for selection: CV validation vs. held-out test.
        "test_vs_validation_gap_pct": criteria["validation_score_pct"] - test_primary_pct,
    }

    # Everything else (names, per-model comparison, per-class reports).
    details = {
        "champion_model": champion_name,
        "primary_metric": metric_name,
        "selection_basis": (
            f"automated multi-criteria selection on {metric_name}: score gap strictly "
            f"below {policy['max_score_gap_pct (strictly below)']}%, then highest "
            "validation score (within tolerance), then smaller model size, lower "
            "training time and lower prediction time"
        ),
        "selection_explanation": champion_info["explanation"],
        "selection_policy": policy,
        "selection_comparison": {
            name: {
                k: selection[name][k]
                for k in (
                    "model_size_mb",
                    "training_time_seconds",
                    "prediction_time_ms",
                    "training_score_pct",
                    "validation_score_pct",
                    "score_gap_pct",
                    "eligible",
                    "status",
                )
            }
            for name in MODEL_NAMES
        },
        "all_models_test_metrics": all_results,
        "champion_confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "champion_classification_report": detailed_report(champion, X_test, y_test),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    with open(DETAILS_PATH, "w", encoding="utf-8") as file:
        json.dump(details, file, indent=4)

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name="champion_evaluation"):
        mlflow.log_param("champion_model", champion_name)
        mlflow.log_param("primary_metric", metric_name)
        mlflow.log_param("selection_method", "multi_criteria_gap_score_size_time")
        mlflow.log_metrics({
            "test_accuracy": champion_metrics["accuracy"],
            "test_balanced_accuracy": champion_metrics["balanced_accuracy"],
            "test_precision": champion_metrics["precision"],
            "test_recall": champion_metrics["recall"],
            "test_f1": champion_metrics["f1"],
            "test_roc_auc": champion_metrics["roc_auc"],
            "test_pr_auc": champion_metrics["pr_auc"],
            "champion_training_score_pct": criteria["training_score_pct"],
            "champion_validation_score_pct": criteria["validation_score_pct"],
            "champion_score_gap_pct": criteria["score_gap_pct"],
            "champion_model_size_mb": criteria["model_size_mb"],
            "champion_training_time_seconds": criteria["training_time_seconds"],
            "champion_prediction_time_ms": criteria["prediction_time_ms"],
        })
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(DETAILS_PATH))
        mlflow.log_artifact(str(SELECTION_PATH))
        if COMPARISON_CSV_PATH.exists():
            mlflow.log_artifact(str(COMPARISON_CSV_PATH))

    print(f"Champion model: {champion_name}")
    print(f"Why: {champion_info['explanation']}")
    print(f"Validation {metric_name}: {criteria['validation_score_pct']:.2f}%  "
          f"Gap: {criteria['score_gap_pct']:.2f}pp")
    print(f"Test F1: {champion_metrics['f1']:.4f}  "
          f"ROC-AUC: {champion_metrics['roc_auc']:.4f}  "
          f"PR-AUC: {champion_metrics['pr_auc']:.4f}")
    print(f"Test precision: {champion_metrics['precision']:.4f}  "
          f"recall: {champion_metrics['recall']:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
