import pandas as pd
from pathlib import Path


def monitor_predictions():

    log_path = Path("logs/predictions.csv")

    if not log_path.exists():
        print("No prediction logs found.")
        return

    df = pd.read_csv(log_path)

    total_predictions = len(df)

    prediction_counts = df["prediction"].value_counts()

    average_probability = df[
        "subscription_probability"
    ].mean()

    print("\n===== PREDICTION MONITORING REPORT =====")

    print(f"\nTotal Predictions: {total_predictions}")

    print("\nPrediction Distribution:")
    print(prediction_counts)

    print(
        f"\nAverage Subscription Probability: "
        f"{average_probability:.4f}"
    )

    print("\n========================================")


if __name__ == "__main__":
    monitor_predictions()