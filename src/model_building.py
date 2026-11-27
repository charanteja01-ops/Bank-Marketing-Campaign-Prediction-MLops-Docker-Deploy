"""Model building with automated multi-criteria champion selection
(Bank Marketing Campaign Prediction).

Step 1  Train all candidate models (Optuna search + stratified CV per model).
Step 2  Measure six criteria per model:
        model size (MB), training time (s), prediction time (ms),
        training score (%), validation score (%), score gap (pp).
Step 3  Apply selection rules:
        - score gap must be strictly below the configured threshold
        - higher validation score preferred
        - smaller size / lower training time / lower prediction time preferred
Step 4  Select the champion automatically and write reports/model_selection.json.

The test set is NEVER read here. Final evaluation happens in model_evaluation.py.
"""
from pathlib import Path
import json
import os
import shutil
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_TRAIN_PATH = Path("data/processed/X_train.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SELECTION_PATH = REPORTS_DIR / "model_selection.json"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"
COMPARISON_MD_PATH = REPORTS_DIR / "model_comparison.md"

MODEL_NAMES = [
    "KNN",
    "DecisionTree",
    "LogisticRegression",
    "GaussianNB",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
]
DISPLAY_NAMES = {
    "KNN": "KNN",
    "DecisionTree": "Decision Tree",
    "LogisticRegression": "Logistic Regression",
    "GaussianNB": "Gaussian NB",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "XGBoost": "XGBoost",
}


def suggest_params(model_name, trial, neg_pos_ratio=1.0):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])

    if model_name == "KNN":
        metric = trial.suggest_categorical("metric", ["minkowski", "manhattan", "chebyshev"])
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 61, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": metric,
            "p": trial.suggest_int("p", 1, 3) if metric == "minkowski" else 2,
            "leaf_size": trial.suggest_int("leaf_size", 15, 60, step=5),
        }
    elif model_name == "DecisionTree":
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 2, 35),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 60),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.02),
        }
    elif model_name == "LogisticRegression":
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
            "solver": solver,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        if solver == "saga":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    elif model_name == "GaussianNB":
        params = {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True)
        }
    elif model_name == "RandomForest":
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 60),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": bootstrap,
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0) if bootstrap else None,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]) if bootstrap else None,
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.01),
        }
    elif model_name == "GradientBoosting":
        params = {
            "loss": trial.suggest_categorical("loss", ["log_loss", "exponential"]),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 60),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.01),
        }
    elif model_name == "XGBoost":
        sqrt_ratio = round(float(np.sqrt(max(neg_pos_ratio, 1.0))), 2)
        full_ratio = round(float(neg_pos_ratio), 2)
        ratio_choices = sorted({1.0, sqrt_ratio, full_ratio})
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 25.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", ratio_choices),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return scaler_type, params


def build_preprocessor(scaler_type):
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
    ])


def build_pipeline(model_name, params, scaler_type):
    if model_name == "KNN":
        estimator = KNeighborsClassifier(**params, n_jobs=-1)
    elif model_name == "DecisionTree":
        estimator = DecisionTreeClassifier(**params, random_state=42)
    elif model_name == "LogisticRegression":
        estimator = LogisticRegression(**params, max_iter=3000, random_state=42)
    elif model_name == "GaussianNB":
        estimator = GaussianNB(**params)
    elif model_name == "RandomForest":
        estimator = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif model_name == "GradientBoosting":
        estimator = GradientBoostingClassifier(**params, random_state=42)
    elif model_name == "XGBoost":
        estimator = XGBClassifier(
            **params,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("Preprocessor", build_preprocessor(scaler_type)),
        ("Model", estimator),
    ])


def fit_with_timing(model_name, params, scaler_type, X, y, repeats):
    times = []
    pipeline = None
    for _ in range(max(1, repeats)):
        pipeline = build_pipeline(model_name, params, scaler_type)
        start = time.perf_counter()
        pipeline.fit(X, y)
        times.append(time.perf_counter() - start)
    return pipeline, float(np.median(times))


def measure_prediction_ms(model, X, n_samples, repeats, seed):
    sample = X.sample(n=min(n_samples, len(X)), random_state=seed)
    rows = [sample.iloc[[i]] for i in range(len(sample))]

    for row in rows[:3]:
        model.predict(row)

    per_repeat_ms = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        for row in rows:
            model.predict(row)
        per_repeat_ms.append((time.perf_counter() - start) / len(rows) * 1000.0)
    return float(np.median(per_repeat_ms))


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def _normalise(values):
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def select_champion(results, cfg):
    gap_limit = float(cfg["max_score_gap_pct"])
    tolerance = float(cfg["score_tolerance_pct"])
    weights = cfg["weights"]

    for r in results.values():
        r["eligible"] = bool(r["score_gap_pct"] < gap_limit)

    eligible = {n: r for n, r in results.items() if r["eligible"]}
    fallback_used = False

    if not eligible:
        mode = cfg.get("fallback_if_none_eligible", "smallest_gap")
        if mode == "error":
            raise RuntimeError(
                f"No model has a score gap strictly below {gap_limit}%."
            )
        warnings.warn("No eligible model; falling back to smallest score gap.")
        fallback_used = True
        eligible = {n: r for n, r in results.items()}

    best_val = max(r["validation_score_pct"] for r in eligible.values())
    if fallback_used:
        smallest_gap = min(r["score_gap_pct"] for r in eligible.values())
        finalists = {
            n: r for n, r in eligible.items()
            if r["score_gap_pct"] - smallest_gap < 1e-9
        }
    else:
        finalists = {
            n: r for n, r in eligible.items()
            if best_val - r["validation_score_pct"] <= tolerance + 1e-9
        }

    norm_size = _normalise({n: r["model_size_mb"] for n, r in finalists.items()})
    norm_train = _normalise({n: r["training_time_seconds"] for n, r in finalists.items()})
    norm_pred = _normalise({n: r["prediction_time_ms"] for n, r in finalists.items()})

    for n, r in finalists.items():
        r["selection_cost"] = float(
            weights["model_size_mb"] * norm_size[n]
            + weights["training_time_seconds"] * norm_train[n]
            + weights["prediction_time_ms"] * norm_pred[n]
        )

    ranked = sorted(
        finalists,
        key=lambda n: (
            round(finalists[n]["selection_cost"], 12),
            -finalists[n]["validation_score_pct"],
            n,
        ),
    )
    champion = ranked[0]

    for n, r in results.items():
        if n == champion:
            r["status"] = "champion"
        elif n in finalists:
            r["status"] = "finalist_runner_up"
        elif r["eligible"]:
            r["status"] = "eligible_but_below_score_tolerance"
        else:
            r["status"] = f"rejected_score_gap_>={gap_limit}%"

    for rank, n in enumerate(ranked, start=1):
        results[n]["finalist_rank"] = rank

    rule_text = (
        "no model met the gap rule, so smallest gap was used"
        if fallback_used
        else f"passed gap rule (< {gap_limit}%)"
    )
    explanation = (
        f"{DISPLAY_NAMES[champion]} selected: {rule_text}; "
        f"{len(finalists)} finalist(s) within {tolerance} pp of best validation "
        f"score ({best_val:.2f}%); lowest weighted size/train-time/predict-time cost."
    )
    return champion, explanation, fallback_used, ranked


def write_comparison_reports(results, champion, metric_name):
    rows = []
    for name in MODEL_NAMES:
        r = results[name]
        rows.append({
            "Model": DISPLAY_NAMES[name],
            "Size MB": round(r["model_size_mb"], 4),
            "Train time s": round(r["training_time_seconds"], 4),
            "Predict ms": round(r["prediction_time_ms"], 4),
            f"Train {metric_name} %": round(r["training_score_pct"], 2),
            f"Val {metric_name} %": round(r["validation_score_pct"], 2),
            "Gap %": round(r["score_gap_pct"], 2),
            "Eligible": "Yes" if r["eligible"] else "No",
            "Status": r["status"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_CSV_PATH, index=False)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    md = "\n".join(
        ["# Model Comparison", "", f"Primary metric: **{metric_name}**", "",
         header, sep, *body, "",
         f"**Champion:** {DISPLAY_NAMES[champion]}", ""]
    )
    COMPARISON_MD_PATH.write_text(md, encoding="utf-8")
    return df


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    target = config["data"]["target"]
    metric_name = config["metric"]["primary"]
    scorer = get_scorer(metric_name)

    sel_cfg = config["selection"]
    meas_cfg = config["measurement"]
    gap_limit = float(sel_cfg["max_score_gap_pct"])
    use_penalty = bool(sel_cfg.get("gap_penalty_in_search", True))
    penalty_weight = float(sel_cfg.get("gap_penalty_weight", 1.0))

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[target].astype(int)

    pos_count = max(int((y_train == 1).sum()), 1)
    neg_count = max(int((y_train == 0).sum()), 1)
    neg_pos_ratio = neg_count / pos_count

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    random_state = int(config["random_state"])
    cv_folds = int(config["cv_folds"])
    n_trials = int(config["n_trials"])

    mlflow.set_experiment(config["experiment_name"])

    results = {}

    for model_name in MODEL_NAMES:
        print(f"\n--- Optimizing {model_name} ---")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("primary_metric", metric_name)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("max_score_gap_pct", gap_limit)

            def objective(trial):
                scaler_type, model_params = suggest_params(
                    model_name, trial, neg_pos_ratio=neg_pos_ratio
                )
                candidate = build_pipeline(model_name, model_params, scaler_type)
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=random_state
                )
                cv_result = cross_validate(
                    candidate, X_train, y_train, scoring=metric_name, cv=cv,
                    n_jobs=1, return_train_score=True,
                )
                val_score = float(cv_result["test_score"].mean())
                train_score = float(cv_result["train_score"].mean())
                gap_pct = (train_score - val_score) * 100.0

                objective_value = val_score
                if use_penalty and gap_pct >= gap_limit:
                    objective_value -= penalty_weight * (gap_pct - gap_limit + 1.0) / 100.0

                trial.set_user_attr("cv_validation_score", val_score)
                trial.set_user_attr("cv_training_score", train_score)
                trial.set_user_attr("cv_gap_pct", gap_pct)

                with mlflow.start_run(
                    run_name=f"{model_name}_trial_{trial.number}", nested=True
                ):
                    mlflow.log_param("scaler_type", scaler_type)
                    mlflow.log_params(model_params)
                    mlflow.log_metric(f"cv_{metric_name}", val_score)
                    mlflow.log_metric(f"cv_train_{metric_name}", train_score)
                    mlflow.log_metric("cv_gap_pct", gap_pct)
                    mlflow.log_metric("objective_value", float(objective_value))

                return float(objective_value)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            search_start = time.time()
            study.optimize(objective, n_trials=n_trials)
            search_time = time.time() - search_start

            best_scaler = study.best_params["scaler_type"]
            best_model_params = {
                k: v for k, v in study.best_params.items() if k != "scaler_type"
            }
            best_cv_val_score = float(study.best_trial.user_attrs["cv_validation_score"])

            final_pipeline, train_time = fit_with_timing(
                model_name, best_model_params, best_scaler, X_train, y_train,
                int(meas_cfg["training_time_repeats"]),
            )

            model_path = MODELS_DIR / f"{model_name}.pkl"
            joblib.dump(final_pipeline, model_path)
            size_mb = file_size_mb(model_path)

            train_score_pct = 100.0 * float(scorer(final_pipeline, X_train, y_train))
            val_score_pct = 100.0 * best_cv_val_score
            gap_pct = train_score_pct - val_score_pct

            pred_ms = measure_prediction_ms(
                final_pipeline, X_train,
                int(meas_cfg["prediction_n_samples"]),
                int(meas_cfg["prediction_repeats"]),
                random_state,
            )

            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metrics({
                f"best_cv_{metric_name}": best_cv_val_score,
                "training_score_pct": train_score_pct,
                "validation_score_pct": val_score_pct,
                "score_gap_pct": gap_pct,
                "model_size_mb": size_mb,
                "training_time_seconds": train_time,
                "prediction_time_ms": pred_ms,
                "search_time_seconds": float(search_time),
            })
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.sklearn.log_model(
                final_pipeline,
                name=f"{model_name}_bank_model",
                serialization_format="cloudpickle",
            )

            results[model_name] = {
                "best_params": study.best_params,
                "model_size_mb": float(size_mb),
                "training_time_seconds": float(train_time),
                "prediction_time_ms": float(pred_ms),
                "training_score_pct": float(train_score_pct),
                "validation_score_pct": float(val_score_pct),
                "score_gap_pct": float(gap_pct),
                "search_time_seconds": float(search_time),
                "model_path": str(model_path).replace("\\", "/"),
            }

            print(
                f"{model_name}: val_{metric_name}={val_score_pct:.2f}% "
                f"train={train_score_pct:.2f}% gap={gap_pct:.2f}pp "
                f"size={size_mb:.3f}MB fit={train_time:.3f}s predict={pred_ms:.3f}ms"
            )

    champion, explanation, fallback_used, ranked = select_champion(results, sel_cfg)

    champion_path = MODELS_DIR / "model.pkl"
    shutil.copy2(MODELS_DIR / f"{champion}.pkl", champion_path)

    champ = results[champion]
    selection = dict(results)
    selection["champion"] = {
        "model_name": champion,
        "best_cv_score": champ["validation_score_pct"] / 100.0,
        "display_name": DISPLAY_NAMES[champion],
        "criteria": {
            "model_size_mb": champ["model_size_mb"],
            "training_time_seconds": champ["training_time_seconds"],
            "prediction_time_ms": champ["prediction_time_ms"],
            "training_score_pct": champ["training_score_pct"],
            "validation_score_pct": champ["validation_score_pct"],
            "score_gap_pct": champ["score_gap_pct"],
        },
        "explanation": explanation,
    }
    selection["champion_model_path"] = str(champion_path).replace("\\", "/")
    selection["primary_metric"] = metric_name
    selection["selection_policy"] = {
        "max_score_gap_pct (strictly below)": gap_limit,
        "score_tolerance_pct": float(sel_cfg["score_tolerance_pct"]),
        "weights": sel_cfg["weights"],
        "fallback_used": fallback_used,
        "finalist_ranking": ranked,
        "selection_data": "training data only (CV validation); test set untouched",
    }

    with open(SELECTION_PATH, "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=4)

    comparison = write_comparison_reports(results, champion, metric_name)

    with mlflow.start_run(run_name="champion_selection"):
        mlflow.log_param("champion_model", champion)
        mlflow.log_param("primary_metric", metric_name)
        mlflow.log_param("fallback_used", fallback_used)
        mlflow.log_metrics({
            "champion_validation_score_pct": champ["validation_score_pct"],
            "champion_score_gap_pct": champ["score_gap_pct"],
            "champion_model_size_mb": champ["model_size_mb"],
            "champion_training_time_seconds": champ["training_time_seconds"],
            "champion_prediction_time_ms": champ["prediction_time_ms"],
        })
        mlflow.log_artifact(str(SELECTION_PATH))
        mlflow.log_artifact(str(COMPARISON_CSV_PATH))
        mlflow.log_artifact(str(COMPARISON_MD_PATH))

    print("\n=== Model comparison ===")
    print(comparison.drop(columns=["Status"]).to_string(index=False))
    print(f"\nChampion: {DISPLAY_NAMES[champion]}")
    print(explanation)
    print(f"Champion model saved to: {champion_path}")
    print(f"Selection details: {SELECTION_PATH}")
    print(f"Comparison report: {COMPARISON_CSV_PATH} / {COMPARISON_MD_PATH}")


if __name__ == "__main__":
    main()