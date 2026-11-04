
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def detect_data_drift(
    reference_path="data/processed/train.csv",
    current_path="data/processed/test.csv",
    report_path="reports/monitoring/data_drift_report.html",
):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # Exclude the target variable from feature drift analysis
    reference = reference.drop(columns=["y"], errors="ignore")
    current = current.drop(columns=["y"], errors="ignore")

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
    print(f"Current dataset rows: {len(current)}")
    print(f"Features monitored: {len(reference.columns)}")
    print(f"HTML report saved to: {output_path}")
    print("=============================")


if __name__ == "__main__":
    detect_data_drift()
