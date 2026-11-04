
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model_performance(
    prediction_path="logs/predictions.csv",
    feedback_path="logs/actual_outcomes.csv",
):
    prediction_path = Path(prediction_path)
    feedback_path = Path(feedback_path)

    if not prediction_path.exists():
        print("Prediction logs not found.")
        return

    if not feedback_path.exists():
        print("Actual outcome logs not found.")
        return

    predictions = pd.read_csv(prediction_path)
    feedback = pd.read_csv(feedback_path)

    required_prediction_columns = {
        "prediction_id",
        "prediction",
        "subscription_probability",
    }

    required_feedback_columns = {
        "prediction_id",
        "actual_outcome",
    }

    if not required_prediction_columns.issubset(predictions.columns):
        print("Required prediction columns are missing.")
        return

    if not required_feedback_columns.issubset(feedback.columns):
        print("Required feedback columns are missing.")
        return

    # Match predictions with their actual outcomes
    merged = predictions.merge(
        feedback,
        on="prediction_id",
        how="inner",
    )

    if merged.empty:
        print("No matched prediction and outcome records found.")
        return

    # Keep valid yes/no labels only
    merged = merged[
        merged["prediction"].isin(["yes", "no"])
        & merged["actual_outcome"].isin(["yes", "no"])
    ].copy()

    if merged.empty:
        print("No valid matched records available.")
        return

    y_true = merged["actual_outcome"]
    y_pred = merged["prediction"]

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, pos_label="yes", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, pos_label="yes", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, pos_label="yes", zero_division=0
    )

    # ROC-AUC requires both actual classes
    if y_true.nunique() == 2:
        y_true_binary = y_true.map({"no": 0, "yes": 1})

        roc_auc = roc_auc_score(
            y_true_binary,
            merged["subscription_probability"],
        )
    else:
        roc_auc = None

    print("\n===== MODEL PERFORMANCE REPORT =====")
    print(f"Evaluated Records: {len(merged)}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC:           {roc_auc:.4f}")
    else:
        print("ROC-AUC:           Not available (only one actual class)")

    print("====================================")


if __name__ == "__main__":
    evaluate_model_performance()
