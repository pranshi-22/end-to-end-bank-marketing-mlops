import pandas as pd
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


def evaluate_models():
    """Evaluate trained models on the test dataset."""

    test_df = pd.read_csv("data/processed/test.csv")

    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]

    model_names = [
        "logistic_regression",
        "random_forest"
    ]

    output_dir = Path("reports/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for name in model_names:
        model = joblib.load(f"models/{name}.joblib")

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        print(f"\n{'=' * 50}")
        print(f"MODEL: {name.upper()}")
        print(f"{'=' * 50}")

        print(classification_report(y_test, predictions))

        auc = roc_auc_score(y_test, probabilities)
        print(f"ROC-AUC: {auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        display = ConfusionMatrixDisplay(confusion_matrix=cm)

        display.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(output_dir / f"{name}_confusion_matrix.png")
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig(output_dir / "roc_curve_comparison.png")
    plt.close()

    print("\nEvaluation completed!")
    print(f"Reports saved in: {output_dir}")


if __name__ == "__main__":
    evaluate_models()