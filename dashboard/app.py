
from pathlib import Path

import pandas as pd
import streamlit as st


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_PATH = PROJECT_ROOT / "logs" / "predictions.csv"
FEEDBACK_PATH = PROJECT_ROOT / "logs" / "actual_outcomes.csv"
DRIFT_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "monitoring" / "data_drift_report.html"
)


# Page configuration
st.set_page_config(
    page_title="Bank Marketing MLOps Dashboard",
    page_icon="📊",
    layout="wide",
)


# Dashboard header
st.title("📊 Bank Marketing MLOps Dashboard")
st.caption("Prediction monitoring and model performance overview")

st.divider()


# Load prediction logs
if PREDICTION_PATH.exists():
    predictions = pd.read_csv(PREDICTION_PATH)
else:
    predictions = pd.DataFrame()

# Load actual outcome logs
if FEEDBACK_PATH.exists():
    feedback = pd.read_csv(FEEDBACK_PATH)
else:
    feedback = pd.DataFrame()


# Prediction statistics
st.subheader("Prediction Monitoring")

if not predictions.empty and "prediction" in predictions.columns:

    total_predictions = len(predictions)

    yes_predictions = (predictions["prediction"] == "yes").sum()
    no_predictions = (predictions["prediction"] == "no").sum()

    average_probability = (
        predictions["subscription_probability"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total_predictions)
    col2.metric("Predicted Yes", int(yes_predictions))
    col3.metric("Predicted No", int(no_predictions))
    col4.metric(
        "Average Subscription Probability",
        f"{average_probability:.2%}",
    )

    st.subheader("Prediction Distribution")

    distribution = predictions["prediction"].value_counts()

    st.bar_chart(distribution)

else:
    st.info("No prediction records available yet.")


st.divider()


# Feedback statistics
st.subheader("Actual Outcome Monitoring")

if not feedback.empty:
    col1, col2 = st.columns(2)

    col1.metric("Feedback Records", len(feedback))

    if "actual_outcome" in feedback.columns:
        col2.metric(
            "Actual Subscriptions",
            int((feedback["actual_outcome"] == "yes").sum()),
        )
else:
    st.info("No actual outcome records available yet.")


# Model performance monitoring
st.divider()

st.subheader("Model Performance Monitoring")

if not predictions.empty and not feedback.empty:

    matched = predictions.merge(
        feedback,
        on="prediction_id",
        how="inner",
    )

    matched = matched[
        matched["prediction"].isin(["yes", "no"])
        & matched["actual_outcome"].isin(["yes", "no"])
    ]

    st.metric("Matched Records", len(matched))

    if not matched.empty:

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        y_true = matched["actual_outcome"]
        y_pred = matched["prediction"]

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

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")
        col4.metric("F1 Score", f"{f1:.2%}")

        if y_true.nunique() == 2:
            y_true_binary = y_true.map({"no": 0, "yes": 1})

            roc_auc = roc_auc_score(
                y_true_binary,
                matched["subscription_probability"],
            )

            st.metric("ROC-AUC", f"{roc_auc:.4f}")
        else:
            st.info(
                "ROC-AUC unavailable: both actual classes "
                "are required."
            )

        if len(matched) < 30:
            st.warning(
                "Small evaluation sample. Collect at least "
                "30 labeled records before interpreting "
                "these metrics."
            )

else:
    st.info("Waiting for matched prediction and feedback records.")


st.divider()


# Data drift report
st.subheader("Data Drift Monitoring")

if DRIFT_REPORT_PATH.exists():
    st.success("Data drift report is available.")
    st.caption(str(DRIFT_REPORT_PATH))
else:
    st.warning("No data drift report found yet.")


st.divider()


# Monitoring Alerts
st.divider()

st.subheader("🚨 Monitoring Alerts")

alerts = []

if len(predictions) < 100:
    alerts.append(
        ("warning", "Low prediction volume: fewer than 100 predictions recorded.")
    )

if len(feedback) < 30:
    alerts.append(
        ("warning", "Insufficient labeled records for reliable model evaluation.")
    )

if not predictions.empty and not feedback.empty:
    if not matched.empty and accuracy < 0.70:
        alerts.append(
            ("error", "Model accuracy is below the 70% threshold.")
        )

if not DRIFT_REPORT_PATH.exists():
    alerts.append(
        ("error", "Data drift report is missing.")
    )

if alerts:
    for alert_type, message in alerts:
        if alert_type == "error":
            st.error(message)
        else:
            st.warning(message)
else:
    st.success("No monitoring alerts. All configured checks passed.")


st.caption("Bank Marketing Subscription Prediction | MLOps Project")
