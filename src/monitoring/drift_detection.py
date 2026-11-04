
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


MIN_CURRENT_ROWS = 100


def detect_data_drift(
    reference_path="data/processed/train.csv",
    current_path="logs/predictions.csv",
    report_path="reports/monitoring/data_drift_report.html",
):
    reference_path = Path(reference_path)
    current_path = Path(current_path)

    if not reference_path.exists():
        print(f"Reference dataset not found: {reference_path}")
        return

    if not current_path.exists():
        print("No prediction logs found. Drift detection skipped.")
        return

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # Remove target and prediction metadata from reference data
    reference = reference.drop(columns=["y"], errors="ignore")

    # Use only the customer input features in prediction logs
    current = current.drop(
        columns=[
            "y",
            "prediction",
            "subscription_probability",
            "timestamp",
        ],
        errors="ignore",
    )

    # Ensure both datasets contain the same features in the same order
    missing_features = [
        column for column in reference.columns
        if column not in current.columns
    ]

    if missing_features:
        print(f"Missing features in prediction logs: {missing_features}")
        print("Drift detection skipped.")
        return

    current = current[reference.columns]

    if len(current) < MIN_CURRENT_ROWS:
        print("\n===== DATA DRIFT DETECTION =====")
        print(f"Current prediction records: {len(current)}")
        print(f"Minimum required records: {MIN_CURRENT_ROWS}")
        print("Insufficient prediction data. Drift detection skipped.")
        print("================================")
        return

    report = Report(
        metrics=[DataDriftPreset()]
    )

    result = report.run(
        reference_data=reference,
        current_data=current,
    )

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.save_html(str(output_path))

    print("\n===== DATA DRIFT REPORT =====")
    print(f"Reference dataset rows: {len(reference)}")
    print(f"Current prediction rows: {len(current)}")
    print(f"Features monitored: {len(reference.columns)}")
    print(f"HTML report saved to: {output_path}")
    print("=============================")


if __name__ == "__main__":
    detect_data_drift()
