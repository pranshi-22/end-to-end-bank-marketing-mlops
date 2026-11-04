from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
from uuid import uuid4
import pandas as pd
import mlflow.sklearn

app = FastAPI(title="Bank Marketing Prediction API")

model_path = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "champion_model"
)

model = mlflow.sklearn.load_model(str(model_path))

class PredictionRequest(BaseModel):
    customer_data: dict[str, Any]

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_outcome: str

@app.get("/")
def home():
    return {"message": "Bank Marketing Prediction API is running!"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = request.customer_data
        prediction_id = str(uuid4())

        expected_columns = list(
            model.named_steps["preprocessor"].feature_names_in_
        )

        missing = set(expected_columns) - set(input_data.keys())

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {sorted(missing)}"
            )

        input_df = pd.DataFrame(
            [{column: input_data[column] for column in expected_columns}]
        )

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        # Log prediction for future monitoring
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / "predictions.csv"

        log_row = input_df.copy()
        log_row["prediction_id"] = prediction_id
        log_row["prediction"] = "yes" if prediction == 1 else "no"
        log_row["subscription_probability"] = round(probability, 4)
        log_row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

        log_row.to_csv(
            log_path,
            mode="a",
            header=not log_path.exists(),
            index=False
        )

        return {
            "prediction_id": prediction_id,
            "prediction": "yes" if prediction == 1 else "no",
            "subscription_probability": round(probability, 4)
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    try:
        if request.actual_outcome not in ["yes", "no"]:
            raise HTTPException(
                status_code=400,
                detail="actual_outcome must be 'yes' or 'no'"
            )

        log_path = Path("logs/predictions.csv")

        if not log_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No prediction logs found"
            )

        predictions = pd.read_csv(log_path)

        matching_rows = predictions[
            predictions["prediction_id"] == request.prediction_id
        ]

        if matching_rows.empty:
            raise HTTPException(
                status_code=404,
                detail="Prediction ID not found"
            )

        feedback_path = Path("logs/actual_outcomes.csv")

        if feedback_path.exists():
            feedback = pd.read_csv(feedback_path)

            if request.prediction_id in feedback["prediction_id"].values:
                raise HTTPException(
                    status_code=409,
                    detail="Feedback already submitted for this prediction"
                )

        feedback_row = pd.DataFrame([{
            "prediction_id": request.prediction_id,
            "actual_outcome": request.actual_outcome,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }])

        feedback_row.to_csv(
            feedback_path,
            mode="a",
            header=not feedback_path.exists(),
            index=False
        )

        return {
            "message": "Feedback recorded successfully",
            "prediction_id": request.prediction_id,
            "actual_outcome": request.actual_outcome
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))