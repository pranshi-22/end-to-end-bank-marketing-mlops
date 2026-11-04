
import pandas as pd

from src.monitoring.monitor_predictions import monitor_predictions


def test_monitor_predictions_runs(tmp_path, capsys):
    log_path = tmp_path / "predictions.csv"

    sample_data = pd.DataFrame({
        "prediction": ["yes", "no", "yes"],
        "subscription_probability": [0.8, 0.3, 0.9],
    })

    sample_data.to_csv(log_path, index=False)

    monitor_predictions(
        log_path=log_path,
        run_drift=False,
    )

    captured = capsys.readouterr()

    assert "Total Predictions: 3" in captured.out
    assert "yes" in captured.out
    assert "no" in captured.out
    assert "0.6667" in captured.out


def test_monitor_predictions_without_logs(tmp_path, capsys):
    log_path = tmp_path / "missing.csv"

    monitor_predictions(
        log_path=log_path,
        run_drift=False,
    )

    captured = capsys.readouterr()

    assert "No prediction logs found." in captured.out
