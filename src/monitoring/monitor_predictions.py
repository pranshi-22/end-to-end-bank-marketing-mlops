
from pathlib import Path

import pandas as pd

from src.monitoring.drift_detection import detect_data_drift


def monitor_predictions(log_path="logs/predictions.csv", run_drift=True):

    log_path = Path(log_path)

    if not log_path.exists():
        print("No prediction logs found.")
    else:
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

        if run_drift:
            print("\nRunning data drift detection...")
            detect_data_drift()


if __name__ == "__main__":
    monitor_predictions()
