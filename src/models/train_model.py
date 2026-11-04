import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def train_models():
    """Train and evaluate baseline classification models."""

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop(columns=["y"])
    y_train = train_df["y"]

    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]

    preprocessor = joblib.load("models/preprocessor.joblib")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities)
        }

        results.append(metrics)

        joblib.dump(pipeline, model_dir / f"{name}.joblib")

        print(f"\n{name}")
        print(classification_report(y_test, predictions))
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_results.csv", index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    train_models()