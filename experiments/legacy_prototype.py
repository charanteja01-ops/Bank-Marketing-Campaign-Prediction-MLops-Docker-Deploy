import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, precision_score, recall_score, roc_auc_score,
)
import joblib
import time

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Put bank.csv in the data/ folder, or set BANK_DATA_PATH to point elsewhere.
DATA_PATH = Path(os.environ.get("BANK_DATA_PATH", "data/bank.csv"))
TARGET = "deposit"
POSITIVE_LABEL = "yes"
RANDOM_STATE = 42
N_TRIALS = 20  # Optuna trials per model (lower this for a quick test)

# Load Dataset
data = pd.read_csv(DATA_PATH)

# Data Cleaning
data = data.drop_duplicates()
# contact: not important | poutcome: ~75% 'unknown'
data = data.drop(columns=["contact", "poutcome"])

# Segregate features and Target
X = data.drop(columns=[TARGET])
y = data[TARGET]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Replace 'unknown' with the most frequent value of the TRAINING data only
X_train, X_test = X_train.copy(), X_test.copy()
for col in ["job", "education"]:
    mode_value = X_train.loc[X_train[col] != "unknown", col].mode()[0]
    X_train[col] = X_train[col].replace("unknown", mode_value)
    X_test[col] = X_test[col].replace("unknown", mode_value)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# F1 score of the positive class (deposit = 'yes')
f1_scorer = make_scorer(f1_score, pos_label=POSITIVE_LABEL)

# Define Pipeline
preprocessor = ColumnTransformer(
    [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("Scaler", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols),
    ]
)

pipeline = Pipeline(
    [
        ('Preprocessor', preprocessor),
        ('Model', KNeighborsClassifier())
    ]
)

SCALER_PARAM = "Preprocessor__num__Scaler"  # where the scaler lives in the pipeline

# KNN Objective
def objective_knn(trial):
  # Define Hyperparameters
  scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
  pipeline.set_params(**{SCALER_PARAM: StandardScaler() if scaler_type == 'standard' else MinMaxScaler()})
  pipeline.set_params(Model = KNeighborsClassifier())
  pipeline.set_params(Model__n_neighbors = trial.suggest_int('n_neighbors', 3, 31, step=2))
  pipeline.set_params(Model__weights = trial.suggest_categorical('weights', ['uniform', 'distance']))
  pipeline.set_params(Model__p = trial.suggest_int('p', 1, 3))
  skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=RANDOM_STATE)
  score = cross_val_score(pipeline, X_train, y_train, scoring=f1_scorer, cv = skf).mean()
  return score

# Logistic Regression Objective
def objective_lr(trial):
  # Define hyperparameters
  scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
  pipeline.set_params(**{SCALER_PARAM: StandardScaler() if scaler_type == 'standard' else MinMaxScaler()})
  pipeline.set_params(
      Model=LogisticRegression(
          C=trial.suggest_float('C', 1e-3, 1e2, log=True),
          solver=trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
          class_weight='balanced',
          max_iter=2000,
          random_state=RANDOM_STATE
      )
  )
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
  score = cross_val_score(pipeline, X_train, y_train, scoring=f1_scorer, cv=skf).mean()
  return score

# Decision Tree Objective
def objective_dt(trial):
  # Define hyperparameters
  scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
  pipeline.set_params(**{SCALER_PARAM: StandardScaler() if scaler_type == 'standard' else MinMaxScaler()})
  pipeline.set_params(Model=DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE))
  pipeline.set_params(
      Model__criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
      Model__max_depth=trial.suggest_int('max_depth', 2, 30),
      Model__min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
      Model__min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
      Model__max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']))
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
  score = cross_val_score(pipeline, X_train, y_train, scoring=f1_scorer, cv=skf).mean()
  return score

# SVM Objective
def objective_svm(trial):
    # Scaler (required for SVM)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(
        **{SCALER_PARAM: StandardScaler() if scaler_type == 'standard' else MinMaxScaler()}
    )

    # Kernel first (controls valid params)
    # (poly / sigmoid are left out: too slow on ~9k rows with one-hot features)
    kernel = trial.suggest_categorical(
        'kernel', ['linear', 'rbf']
    )

    # Base hyperparameters
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e1, log=True),
        'kernel': kernel
    }

    # Kernel-specific hyperparameters
    if kernel == 'rbf':
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)

    # Set model
    pipeline.set_params(
        Model=SVC(**params, class_weight='balanced', random_state=RANDOM_STATE)
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring=f1_scorer,
        cv=skf
    ).mean()
    return score

# GaussianNB Objective
def objective_gnb(trial):
    # Scaler (optional for GNB, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(
        **{SCALER_PARAM: StandardScaler() if scaler_type == 'standard' else MinMaxScaler()}
    )

    # GaussianNB hyperparameters
    pipeline.set_params(
        Model=GaussianNB(
            var_smoothing=trial.suggest_float(
                'var_smoothing', 1e-11, 1e-7, log=True
            )
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring=f1_scorer,
        cv=skf
    ).mean()

    return score

# Random Forest Objective
def objective_rf(trial):
    # Scaler (optional for Random Forest, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(
        **{SCALER_PARAM: StandardScaler() if scaler_type == 'standard' else MinMaxScaler()}
    )

    # Random Forest hyperparameters
    pipeline.set_params(
        Model=RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 400, step=50),
            criterion=trial.suggest_categorical(
                'criterion', ['gini', 'entropy', 'log_loss']
            ),
            max_depth=trial.suggest_int('max_depth', 5, 30),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', None]
            ),
            bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring=f1_scorer,
        cv=skf
    ).mean()

    return score

# Gradient Boosting Objective
def objective_gb(trial):
    # Scaler (optional for Gradient Boosting, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(
        **{SCALER_PARAM: StandardScaler() if scaler_type == 'standard' else MinMaxScaler()}
    )

    # Gradient Boosting hyperparameters
    pipeline.set_params(
        Model=GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 300, step=50),
            learning_rate=trial.suggest_float(
                'learning_rate', 0.01, 0.3, log=True
            ),
            max_depth=trial.suggest_int('max_depth', 2, 6),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', None]
            ),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=42
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring=f1_scorer,
        cv=skf
    ).mean()

    return score

# Map model names to objective functions
objectives = {
    "KNN": objective_knn,
    "LogisticRegression": objective_lr,
    "DecisionTree": objective_dt,
    "SVM": objective_svm,
    "GaussianNB": objective_gnb,
    "RandomForest": objective_rf,
    "GradientBoosting": objective_gb
}


def positive_scores(model, X_data):
    if hasattr(model, "predict_proba"):
        idx = list(model.classes_).index(POSITIVE_LABEL)
        return model.predict_proba(X_data)[:, idx]
    return model.decision_function(X_data)  # SVC without probabilities


# Set experiment
mlflow.set_experiment("Bank_Marketing_PL_RUNS")

results = {}
model_dict = {name: i for i, name in enumerate(objectives.keys())}
scaler_dict = {name: i for i, name in enumerate(['standard', 'minmax'])}

# Loop through each algorithm
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    # One parent run per algorithm; every Optuna trial is a child run of it
    with mlflow.start_run(run_name=model_name):

        mlflow_cb = MLflowCallback(
            tracking_uri=None,              # Auto-detect (str/None)
            metric_name="cv_f1",            # Primary metric (str)
            mlflow_kwargs={                 # **MLflow start_run() kwargs**
                "nested": True              # Child runs under parent
            }
        )

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )

        # Search for the best hyperparameters
        start_search = time.time()
        study.optimize(obj_fn, n_trials=N_TRIALS, callbacks=[mlflow_cb])
        search_time = time.time() - start_search

        print(f"Best CV F1 for {model_name}: {study.best_value:.4f}")
        best_params = study.best_params
        results[model_name] = {"best_params": best_params, "best_cv_f1": study.best_value}

        # Fit the pipeline with the best parameters
        scaler = StandardScaler() if best_params["scaler_type"]=="standard" else MinMaxScaler()
        pipeline.set_params(**{SCALER_PARAM: scaler})

        if model_name == "KNN":
            pipeline.set_params(
                Model=KNeighborsClassifier(),
                Model__n_neighbors=best_params["n_neighbors"],
                Model__weights=best_params["weights"],
                Model__p=best_params["p"]
            )
        elif model_name == "LogisticRegression":
            pipeline.set_params(
                Model=LogisticRegression(
                    C=best_params["C"],
                    solver=best_params["solver"],
                    class_weight='balanced',
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            )
        elif model_name == "DecisionTree":
            pipeline.set_params(
                Model=DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
                Model__criterion=best_params["criterion"],
                Model__max_depth=best_params["max_depth"],
                Model__min_samples_split=best_params["min_samples_split"],
                Model__min_samples_leaf=best_params["min_samples_leaf"],
                Model__max_features=best_params["max_features"]
            )
        elif model_name == "SVM":
            params = {"kernel": best_params["kernel"], "C": best_params["C"]}
            if best_params["kernel"] == "rbf":
                params["gamma"] = best_params["gamma"]
            pipeline.set_params(
                Model=SVC(**params, class_weight='balanced', random_state=RANDOM_STATE)
            )
        elif model_name == "GaussianNB":
            pipeline.set_params(
                Model=GaussianNB(),
                Model__var_smoothing=best_params["var_smoothing"]
            )
        elif model_name == "RandomForest":
            pipeline.set_params(
                Model=RandomForestClassifier(
                    class_weight='balanced', random_state=42, n_jobs=-1
                ),
                Model__n_estimators=best_params["n_estimators"],
                Model__criterion=best_params["criterion"],
                Model__max_depth=best_params["max_depth"],
                Model__min_samples_split=best_params["min_samples_split"],
                Model__min_samples_leaf=best_params["min_samples_leaf"],
                Model__max_features=best_params["max_features"],
                Model__bootstrap=best_params["bootstrap"]
            )
        elif model_name == "GradientBoosting":
            pipeline.set_params(
                Model=GradientBoostingClassifier(random_state=42),
                Model__n_estimators=best_params["n_estimators"],
                Model__learning_rate=best_params["learning_rate"],
                Model__max_depth=best_params["max_depth"],
                Model__min_samples_split=best_params["min_samples_split"],
                Model__min_samples_leaf=best_params["min_samples_leaf"],
                Model__max_features=best_params["max_features"],
                Model__subsample=best_params["subsample"]
            )

        # Train the final model (timed separately from the search)
        start_fit = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_fit

        # Evaluate on test data
        start_test = time.time()
        y_pred = pipeline.predict(X_test)
        test_time = time.time() - start_test
        y_pred_train = pipeline.predict(X_train)

        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train, pos_label=POSITIVE_LABEL)
        test_acc = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        test_recall = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        test_f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        test_auc = roc_auc_score(
            (y_test == POSITIVE_LABEL).astype(int), positive_scores(pipeline, X_test)
        )

        print(f"{model_name} Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")
        print(f"{model_name} Training F1: {train_f1:.4f}, Testing F1: {test_f1:.4f}")
        print(f"{model_name} Fit Time: {fit_time:.2f}s, Test Time: {test_time:.2f}s")

        # Save model manually to track model size
        model_path = f"{model_name}_final_model.pkl"
        joblib.dump(pipeline, model_path)
        model_size = os.path.getsize(model_path)

        mlflow.log_metric("model_id", model_dict[model_name])
        mlflow.log_metric("Scaler_id", scaler_dict[best_params["scaler_type"]])
        mlflow.log_metric("best_cv_f1", study.best_value)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_roc_auc", test_auc)
        mlflow.log_metric("train_time", fit_time)
        mlflow.log_metric("test_time", test_time)
        mlflow.log_metric("search_time", search_time)
        mlflow.log_metric("model_size", model_size)
        mlflow.sklearn.log_model(
            pipeline,
            name=f"{model_name}_bank_model",
            serialization_format="cloudpickle")  # , registered_model_name = f"Bank_{model_name}_Best")
        os.remove(model_path)

    results[model_name].update({
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "fit_time": fit_time,
        "test_time": test_time,
        "search_time": search_time,
        "model_size_bytes": model_size
    })

# Summary
print("\n--- Summary ---")
for model_name, res in results.items():
    print(f"{model_name}: CV F1={res['best_cv_f1']:.4f}, Train F1={res['train_f1']:.4f}, "
          f"Test F1={res['test_f1']:.4f}, Test Acc={res['test_accuracy']:.4f}, "
          f"Fit Time={res['fit_time']:.2f}s, Model Size={res['model_size_bytes']} bytes")
