from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.pipelines import prepare_task_data


# ==========================================
# Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "H2.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "classification"

RANDOM_STATE = 42
CV_FOLDS = 3


# ==========================================
# Metrics
# ==========================================

def evaluate_classifier(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        results["roc_auc"] = roc_auc_score(y_test, probabilities)

    return results


# ==========================================
# Model definitions
# ==========================================

def get_models() -> dict:
    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 5, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=150,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 15],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }


# ==========================================
# Training and tuning
# ==========================================

def run_experiments() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_task_data(df, "classification")
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []
    best_models = {}

    for model_name, (estimator, param_grid) in get_models().items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print("=" * 60)

        baseline_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )

        start = time.perf_counter()
        baseline_pipeline.fit(X_train, y_train)
        baseline_seconds = time.perf_counter() - start

        baseline_metrics = evaluate_classifier(
            baseline_pipeline,
            X_test,
            y_test,
        )

        results.append({
            "model": model_name,
            "version": "baseline",
            "training_seconds": baseline_seconds,
            **baseline_metrics,
        })

        print("Baseline:", baseline_metrics)

        tuned_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )

        search = GridSearchCV(
            estimator=tuned_pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )

        start = time.perf_counter()
        search.fit(X_train, y_train)
        tuning_seconds = time.perf_counter() - start

        tuned_metrics = evaluate_classifier(
            search.best_estimator_,
            X_test,
            y_test,
        )

        results.append({
            "model": model_name,
            "version": "tuned",
            "training_seconds": tuning_seconds,
            "best_params": json.dumps(search.best_params_),
            "cv_best_f1": search.best_score_,
            **tuned_metrics,
        })

        best_models[model_name] = search.best_estimator_

        print("Best parameters:", search.best_params_)
        print("Tuned metrics:", tuned_metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "classification_model_comparison.csv",
        index=False,
    )

    # Save tuned pipelines for later inference.
    # These artifacts are required locally for the live discussion.
    import joblib

    for model_name, model in best_models.items():
        joblib.dump(
            model,
            ARTIFACT_DIR / f"{model_name}_pipeline.joblib",
        )

    print("\nResults saved to:", OUTPUT_DIR)
    print("Pipelines saved to:", ARTIFACT_DIR)

    return results_df


if __name__ == "__main__":
    comparison = run_experiments()
    print("\nFinal comparison:")
    print(comparison.to_string(index=False))