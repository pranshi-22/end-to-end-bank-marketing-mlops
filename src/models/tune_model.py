import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.features.build_features import build_preprocessor


def tune_model():
    """Tune Logistic Regression using cross-validation."""

    train_df = pd.read_csv("data/processed/train.csv")

    X_train = train_df.drop(columns=["y"])
    y_train = train_df["y"]

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ])

    parameter_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": ["balanced", None]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    joblib.dump(search.best_estimator_, output_dir / "tuned_logistic_regression.joblib")

    pd.DataFrame(search.cv_results_).to_csv(
        output_dir / "tuning_results.csv",
        index=False
    )

    print("\nTuning completed!")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation F1-score: {search.best_score_:.4f}")
    print("Tuned model saved successfully!")


if __name__ == "__main__":
    tune_model()