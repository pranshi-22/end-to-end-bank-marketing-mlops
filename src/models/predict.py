import pandas as pd
import joblib


def predict_subscription(input_data: dict):
    """Predict whether a customer will subscribe to a term deposit."""

    model = joblib.load("models/tuned_logistic_regression.joblib")

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "yes" if prediction == 1 else "no"

    return {
        "prediction": result,
        "subscription_probability": round(float(probability), 4)
    }


if __name__ == "__main__":
    test_df = pd.read_csv("data/processed/test.csv")

    sample = test_df.drop(columns=["y"]).iloc[0].to_dict()

    result = predict_subscription(sample)

    print("Prediction:", result["prediction"])
    print("Subscription probability:", result["subscription_probability"])