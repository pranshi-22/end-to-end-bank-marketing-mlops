
import pandas as pd

from src.monitoring.monitor_predictions import monitor_predictions
from src.monitoring.drift_detection import detect_data_drift


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


def test_drift_detection_without_prediction_logs(tmp_path, capsys):
    reference_path = tmp_path / "train.csv"
    current_path = tmp_path / "missing.csv"

    pd.DataFrame({"age": [25, 30], "y": ["yes", "no"]}).to_csv(
        reference_path, index=False
    )

    detect_data_drift(
        reference_path=reference_path,
        current_path=current_path,
    )

    captured = capsys.readouterr()

    assert "No prediction logs found." in captured.out


def test_drift_detection_insufficient_predictions(tmp_path, capsys):
    reference_path = tmp_path / "train.csv"
    current_path = tmp_path / "predictions.csv"

    pd.DataFrame({"age": [25, 30], "y": ["yes", "no"]}).to_csv(
        reference_path, index=False
    )

    pd.DataFrame({"age": [25, 30, 35]}).to_csv(
        current_path, index=False
    )

    detect_data_drift(
        reference_path=reference_path,
        current_path=current_path,
    )

    captured = capsys.readouterr()

    assert "Insufficient prediction data" in captured.out


def test_drift_detection_missing_features(tmp_path, capsys):
    reference_path = tmp_path / "train.csv"
    current_path = tmp_path / "predictions.csv"

    pd.DataFrame({
        "age": [25, 30],
        "job": ["admin.", "technician"],
        "y": ["yes", "no"],
    }).to_csv(reference_path, index=False)

    pd.DataFrame({
        "age": [25, 30] * 50,
    }).to_csv(current_path, index=False)

    detect_data_drift(
        reference_path=reference_path,
        current_path=current_path,
    )

    captured = capsys.readouterr()

    assert "Missing features in prediction logs" in captured.out
    assert "Drift detection skipped." in captured.out



from src.monitoring.model_performance import evaluate_model_performance


def test_model_performance_with_valid_data(tmp_path, capsys):
    prediction_path = tmp_path / "predictions.csv"
    feedback_path = tmp_path / "actual_outcomes.csv"

    pd.DataFrame({
        "prediction_id": ["p1", "p2"],
        "prediction": ["yes", "no"],
        "subscription_probability": [0.9, 0.2],
    }).to_csv(prediction_path, index=False)

    pd.DataFrame({
        "prediction_id": ["p1", "p2"],
        "actual_outcome": ["yes", "no"],
    }).to_csv(feedback_path, index=False)

    evaluate_model_performance(
        prediction_path=prediction_path,
        feedback_path=feedback_path,
    )

    captured = capsys.readouterr()

    assert "Evaluated Records: 2" in captured.out
    assert "Accuracy:          1.0000" in captured.out
    assert "ROC-AUC:           1.0000" in captured.out


def test_model_performance_without_feedback(tmp_path, capsys):
    prediction_path = tmp_path / "predictions.csv"

    pd.DataFrame({
        "prediction_id": ["p1"],
        "prediction": ["yes"],
        "subscription_probability": [0.9],
    }).to_csv(prediction_path, index=False)

    evaluate_model_performance(
        prediction_path=prediction_path,
        feedback_path=tmp_path / "missing.csv",
    )

    captured = capsys.readouterr()

    assert "Actual outcome logs not found." in captured.out
