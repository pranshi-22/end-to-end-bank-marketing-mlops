import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    f1_score
)
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_tuned_model():
    """Evaluate the tuned Logistic Regression model."""

    test_df = pd.read_csv("data/processed/test.csv")

    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]

    model = joblib.load("models/tuned_logistic_regression.joblib")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nTUNED LOGISTIC REGRESSION")
    print("=" * 50)
    print(classification_report(y_test, predictions))

    print(f"F1-score: {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    output_dir = Path("reports/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions)
    )
    display.plot(cmap="Blues")
    plt.title("Tuned Logistic Regression")
    plt.savefig(output_dir / "tuned_logistic_regression_confusion_matrix.png")
    plt.close()

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    evaluate_tuned_model()